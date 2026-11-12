"""
Trading Pipeline 集成测试
"""

import pytest
import asyncio
from datetime import datetime

from agentrt.orchestrator import (
    AgentOrchestrator,
    BaseAgent,
    AgentMessage,
    AgentResult,
    AgentPriority,
    DataSourceAgent,
    SignalAggregatorAgent,
    RiskAssessmentAgent,
    ExecutionAgent,
)
from agentrt.orchestrator.signal_aggregator_agent import SignalType, ConflictResolution
from agentrt.orchestrator.risk_assessment_agent import RiskLevel
from agentrt.orchestrator.execution_agent import OrderSide, OrderStatus
from agentrt.orchestrator.trading_pipeline import TradingPipeline, PipelineMode


class TestBaseAgent:
    """测试基础代理类"""

    @pytest.fixture
    def simple_agent(self):
        """创建简单测试代理"""
        class TestAgent(BaseAgent):
            async def execute(self, context):
                return AgentResult(
                    success=True,
                    data={"test": "result", "input": context.get("input")},
                )

        return TestAgent("test_agent")

    @pytest.mark.asyncio
    async def test_agent_initialization(self, simple_agent):
        """测试代理初始化"""
        assert simple_agent.name == "test_agent"
        assert simple_agent.priority == AgentPriority.NORMAL
        assert simple_agent.id is not None

    @pytest.mark.asyncio
    async def test_agent_execution(self, simple_agent):
        """测试代理执行"""
        await simple_agent.start()

        result = await simple_agent.run({"input": "test_value"})

        assert result.success
        assert result.data["test"] == "result"
        assert result.data["input"] == "test_value"
        assert result.execution_time_ms > 0

        await simple_agent.stop()

    @pytest.mark.asyncio
    async def test_agent_timeout(self):
        """测试代理超时"""
        class SlowAgent(BaseAgent):
            async def execute(self, context):
                await asyncio.sleep(10)  # 超过默认超时
                return AgentResult(success=True)

        agent = SlowAgent("slow_agent", timeout=0.1)
        await agent.start()

        result = await agent.run({})

        assert not result.success
        assert "timed out" in result.error

        await agent.stop()

    @pytest.mark.asyncio
    async def test_agent_metrics(self, simple_agent):
        """测试代理指标"""
        await simple_agent.start()
        await simple_agent.run({})
        await simple_agent.run({})

        metrics = simple_agent.metrics
        assert metrics["tasks_completed"] == 2
        assert metrics["total_execution_time_ms"] > 0

        await simple_agent.stop()


class TestAgentOrchestrator:
    """测试代理协调器"""

    @pytest.fixture
    def orchestrator(self):
        return AgentOrchestrator("test_orchestrator")

    @pytest.fixture
    def sample_agents(self):
        """创建示例代理"""
        class AgentA(BaseAgent):
            async def execute(self, context):
                return AgentResult(success=True, data={"agent": "A"})

        class AgentB(BaseAgent):
            async def execute(self, context):
                return AgentResult(success=True, data={"agent": "B"})

        class AgentC(BaseAgent):
            async def execute(self, context):
                # 依赖AgentA和AgentB的结果
                dep_results = context.get("dependency_results", {})
                return AgentResult(success=True, data={
                    "agent": "C",
                    "dependencies": list(dep_results.keys())
                })

        return {
            "A": AgentA("agent_a"),
            "B": AgentB("agent_b"),
            "C": AgentC("agent_c"),
        }

    @pytest.mark.asyncio
    async def test_register_agent(self, orchestrator, sample_agents):
        """测试代理注册"""
        orchestrator.register_agent(sample_agents["A"])

        assert "agent_a" in [a["name"] for a in orchestrator.list_agents()]
        assert orchestrator.get_agent("agent_a") is not None

    @pytest.mark.asyncio
    async def test_parallel_execution(self, orchestrator, sample_agents):
        """测试并行执行"""
        for agent in sample_agents.values():
            orchestrator.register_agent(agent)

        await orchestrator.start()

        from agentrt.orchestrator.orchestrator import ExecutionPlan, ExecutionMode

        plan = ExecutionPlan(
            mode=ExecutionMode.PARALLEL,
            agents=["agent_a", "agent_b"],
        )

        result = await orchestrator.execute_plan(plan)

        assert result.success
        assert "agent_a" in result.results
        assert "agent_b" in result.results
        assert result.results["agent_a"].data["agent"] == "A"
        assert result.results["agent_b"].data["agent"] == "B"

        await orchestrator.stop()

    @pytest.mark.asyncio
    async def test_dag_execution(self, orchestrator, sample_agents):
        """测试DAG依赖执行"""
        for agent in sample_agents.values():
            orchestrator.register_agent(agent)

        await orchestrator.start()

        from agentrt.orchestrator.orchestrator import ExecutionPlan, ExecutionMode

        plan = ExecutionPlan(
            mode=ExecutionMode.DAG,
            agents=["agent_a", "agent_b", "agent_c"],
            dependencies={
                "agent_a": [],
                "agent_b": [],
                "agent_c": ["agent_a", "agent_b"],  # C依赖A和B
            },
        )

        result = await orchestrator.execute_plan(plan)

        assert result.success
        assert result.results["agent_c"].data["dependencies"] == ["agent_a", "agent_b"]

        await orchestrator.stop()


class TestDataSourceAgent:
    """测试数据源代理"""

    @pytest.fixture
    def data_agent(self):
        agent = DataSourceAgent("test_data_agent")
        agent.configure_default_sources()
        return agent

    @pytest.mark.asyncio
    async def test_sources_without_a_fetcher_refuse_to_fabricate(self, data_agent, monkeypatch):
        """默认情况下，没有 fetcher 的数据源不得返回随机报价。

        此前 _do_fetch 在没有 fetcher 时用 _mock_fetch 生成随机价格，并标上数据源名字
        （例如 source="binance" 的 AAPL），与真实数据无法区分。
        """
        monkeypatch.delenv("ALLOW_MOCK_AGENT_DATA", raising=False)
        await data_agent.start()

        result = await data_agent.run({
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_type": "quote",
        })

        assert not result.success
        assert "Refusing to return fabricated market data" in result.error

        await data_agent.stop()

    @pytest.mark.asyncio
    async def test_fetch_quotes(self, data_agent, monkeypatch):
        """测试获取报价（显式开启 mock 数据的本地开发路径）"""
        monkeypatch.setenv("ALLOW_MOCK_AGENT_DATA", "true")
        await data_agent.start()

        result = await data_agent.run({
            "symbols": ["BTCUSDT", "ETHUSDT"],
            "data_type": "quote",
        })

        assert result.success
        assert "merged" in result.data
        assert len(result.data["merged"]) == 2

        await data_agent.stop()

    @pytest.mark.asyncio
    async def test_source_status(self, data_agent):
        """测试数据源状态"""
        status = data_agent.get_source_status()

        assert "binance" in status
        assert "yahoo_finance" in status
        assert status["binance"]["enabled"]


class TestSignalAggregatorAgent:
    """测试信号聚合代理"""

    @pytest.fixture
    def signal_agent(self):
        agent = SignalAggregatorAgent("test_signal_agent")
        agent.register_strategy("momentum", 1.0)
        agent.register_strategy("mean_reversion", 0.8)
        return agent

    @pytest.mark.asyncio
    async def test_aggregate_signals(self, signal_agent):
        """测试信号聚合"""
        await signal_agent.start()

        signals = [
            {
                "symbol": "BTCUSDT",
                "signal_type": "buy",
                "strength": 0.8,
                "confidence": 0.7,
                "strategy": "momentum",
            },
            {
                "symbol": "BTCUSDT",
                "signal_type": "buy",
                "strength": 0.6,
                "confidence": 0.6,
                "strategy": "mean_reversion",
            },
        ]

        result = await signal_agent.run({"signals": signals})

        assert result.success
        assert "aggregated" in result.data
        assert "BTCUSDT" in result.data["aggregated"]

        agg_signal = result.data["aggregated"]["BTCUSDT"]
        assert agg_signal.signal_type == SignalType.BUY
        assert agg_signal.consensus_score == 1.0  # 都是BUY，完全一致

        await signal_agent.stop()

    @pytest.mark.asyncio
    async def test_conflict_resolution(self, signal_agent):
        """测试冲突解决"""
        await signal_agent.start()

        signals = [
            {
                "symbol": "BTCUSDT",
                "signal_type": "buy",
                "strength": 0.8,
                "confidence": 0.9,
                "strategy": "momentum",
            },
            {
                "symbol": "BTCUSDT",
                "signal_type": "sell",
                "strength": 0.6,
                "confidence": 0.5,
                "strategy": "mean_reversion",
            },
        ]

        result = await signal_agent.run({
            "signals": signals,
            "conflict_resolution": "weighted_average",
        })

        assert result.success
        # BUY信号更强，应该胜出
        agg_signal = result.data["aggregated"]["BTCUSDT"]
        assert agg_signal.signal_type == SignalType.BUY
        assert agg_signal.consensus_score == pytest.approx(0.72 / 0.96)

        await signal_agent.stop()


class TestRiskAssessmentAgent:
    """测试风险评估代理"""

    @pytest.fixture
    def risk_agent(self):
        agent = RiskAssessmentAgent("test_risk_agent")
        agent.set_portfolio({"BTCUSDT": 10000}, 100000)
        return agent

    @pytest.mark.asyncio
    async def test_risk_assessment(self, risk_agent):
        """测试风险评估"""
        await risk_agent.start()

        result = await risk_agent.run({
            "symbol": "ETHUSDT",
            "action": "buy",
            "quantity": 4,
            "price": 2000,
            "market_data": {"volatility": 0.15},
        })

        assert result.success
        assert result.data is not None
        assert result.data.approved  # 应该批准
        assert result.data.risk_score >= 0

        await risk_agent.stop()

    @pytest.mark.asyncio
    async def test_high_risk_rejection(self, risk_agent):
        """测试高风险交易拒绝"""
        await risk_agent.start()

        # 尝试购买超过仓位限制的数量
        result = await risk_agent.run({
            "symbol": "ETHUSDT",
            "action": "buy",
            "quantity": 100,
            "price": 2000,  # 总价值200000，超过组合价值
            "market_data": {"volatility": 0.6},  # 高波动率
        })

        assert result.success
        assert result.data is not None
        # 应该有风险警告
        assert len(result.data.warnings) > 0

        await risk_agent.stop()


class TestExecutionAgent:
    """测试执行代理

    ExecutionAgent 用随机成交率 / 滑点模拟成交，属于合成数据：默认拒绝，
    编排行为的用例通过 ALLOW_MOCK_AGENT_DATA 显式开启。
    """

    @pytest.fixture
    def execution_agent(self):
        return ExecutionAgent("test_execution_agent")

    @pytest.mark.asyncio
    async def test_simulated_fill_is_refused_by_default(self, execution_agent, monkeypatch):
        """默认（未开启 mock）不得返回随机生成的成交"""
        monkeypatch.delenv("ALLOW_MOCK_AGENT_DATA", raising=False)
        await execution_agent.start()
        result = await execution_agent.run({
            "symbol": "BTCUSDT", "side": "buy", "quantity": 1.0, "price": 40000,
            "order_type": "market", "config": {"algorithm": "immediate"},
        })
        assert not result.success
        assert "random" in result.error and "ExecutionTool" in result.error
        assert result.data is None
        await execution_agent.stop()

    @pytest.mark.asyncio
    async def test_immediate_execution(self, execution_agent, monkeypatch):
        """测试立即执行（显式开启 mock 成交）"""
        monkeypatch.setenv("ALLOW_MOCK_AGENT_DATA", "true")
        await execution_agent.start()

        result = await execution_agent.run({
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 1.0,
            "price": 40000,
            "order_type": "market",
            "config": {"algorithm": "immediate"},
        })

        assert result.success
        assert result.data is not None
        assert result.data.filled_quantity > 0
        assert result.data.status in [OrderStatus.FILLED, OrderStatus.PARTIAL]

        await execution_agent.stop()

    @pytest.mark.asyncio
    async def test_twap_execution(self, execution_agent, monkeypatch):
        """测试TWAP执行（显式开启 mock 成交）"""
        monkeypatch.setenv("ALLOW_MOCK_AGENT_DATA", "true")
        await execution_agent.start()

        result = await execution_agent.run({
            "symbol": "BTCUSDT",
            "side": "buy",
            "quantity": 10.0,
            "price": 40000,
            "order_type": "market",
            "config": {"algorithm": "twap", "time_horizon": 5},
        })

        assert result.success
        assert result.data is not None
        assert result.data.num_child_orders > 1  # 应该有多个子订单

        await execution_agent.stop()


class TestTradingPipeline:
    """测试交易流水线

    TradingPipeline 是演示流水线：默认数据源没有 fetcher、信号为随机生成。
    这些用例通过 ALLOW_MOCK_AGENT_DATA 显式开启 mock 数据来覆盖其编排行为；
    默认（未开启）时必须拒绝，见 test_refuses_to_run_on_fabricated_data。
    """

    @pytest.fixture(autouse=True)
    def allow_mock_data(self, monkeypatch):
        monkeypatch.setenv("ALLOW_MOCK_AGENT_DATA", "true")

    @pytest.fixture
    def pipeline(self):
        return TradingPipeline("test_pipeline", PipelineMode.ANALYSIS)

    @pytest.mark.asyncio
    async def test_refuses_to_run_on_fabricated_data(self, pipeline, monkeypatch):
        monkeypatch.delenv("ALLOW_MOCK_AGENT_DATA", raising=False)
        await pipeline.start()
        try:
            result = await pipeline.run(symbols=["BTCUSDT"], mode=PipelineMode.ANALYSIS)
        finally:
            await pipeline.stop()

        assert result.market_data == {}
        assert "Refusing to return fabricated" in result.errors["data_fetch"]
        assert "Refusing to return fabricated trading signals" in result.errors["pipeline"]
        assert not result.signals
        assert not result.execution_reports

    @pytest.mark.asyncio
    async def test_pipeline_analysis_mode(self, pipeline):
        """测试分析模式"""
        await pipeline.start()

        try:
            result = await pipeline.run(
                symbols=["BTCUSDT", "ETHUSDT"],
                mode=PipelineMode.ANALYSIS,
            )

            assert result.mode == PipelineMode.ANALYSIS
            assert len(result.market_data) > 0
            assert len(result.signals) >= 0  # 可能有信号
            # 分析模式不应该有执行报告
            assert len(result.execution_reports) == 0

        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_signal_only_mode(self, pipeline):
        """测试仅信号模式"""
        await pipeline.start()

        try:
            result = await pipeline.run(
                symbols=["BTCUSDT"],
                mode=PipelineMode.SIGNAL_ONLY,
            )

            assert result.mode == PipelineMode.SIGNAL_ONLY
            # 不应该有风险评估和执行
            assert len(result.risk_assessments) == 0
            assert len(result.execution_reports) == 0

        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_with_custom_signals(self, pipeline):
        """测试使用自定义信号"""
        await pipeline.start()

        try:
            custom_signals = [
                {
                    "symbol": "BTCUSDT",
                    "signal_type": "buy",
                    "strength": 0.9,
                    "confidence": 0.85,
                    "strategy": "custom",
                },
            ]

            result = await pipeline.run(
                symbols=["BTCUSDT"],
                signals=custom_signals,
                mode=PipelineMode.ANALYSIS,
            )

            assert "BTCUSDT" in result.signals
            signal = result.signals["BTCUSDT"]
            assert signal.signal_type == SignalType.BUY

        finally:
            await pipeline.stop()

    @pytest.mark.asyncio
    async def test_pipeline_status(self, pipeline):
        """测试流水线状态"""
        await pipeline.start()

        try:
            status = pipeline.get_status()

            assert status["name"] == "test_pipeline"
            assert status["mode"] == "analysis"
            assert "orchestrator" in status
            assert "agents" in status
            assert "data_source" in status["agents"]

        finally:
            await pipeline.stop()


class TestAgentMessaging:
    """测试代理间消息通信"""

    @pytest.mark.asyncio
    async def test_message_broadcast(self):
        """测试消息广播"""
        orchestrator = AgentOrchestrator("test")

        received_messages = []

        class ReceiverAgent(BaseAgent):
            async def execute(self, context):
                return AgentResult(success=True)

        receiver1 = ReceiverAgent("receiver1")
        receiver2 = ReceiverAgent("receiver2")

        # 订阅主题
        receiver1.subscribe("test.topic")
        receiver2.subscribe("test.topic")

        # 记录收到的消息
        async def record_message(msg):
            received_messages.append(msg)

        receiver1.register_handler("test.topic", record_message)
        receiver2.register_handler("test.topic", record_message)

        orchestrator.register_agent(receiver1)
        orchestrator.register_agent(receiver2)

        await orchestrator.start()

        # 广播消息
        await receiver1.broadcast("test.topic", {"data": "test"})

        # 等待消息路由
        await asyncio.sleep(0.2)

        # receiver2 应该收到消息（receiver1 是发送者，不会收到自己的广播）
        assert len(received_messages) >= 1

        await orchestrator.stop()


# 运行测试
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
