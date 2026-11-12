# Skill Engine Documentation

## Overview

The Skill Engine is a dynamic skill discovery and execution system for TraderX. It enables:

1. **Auto-Discovery**: Automatically scans `.claude/skills/` directory to discover available skills
2. **Metadata Parsing**: Extracts skill metadata from YAML front matter in `SKILL.md` files
3. **Flexible Execution**: Supports both Python handlers and external script execution
4. **Sandboxed Environment**: Executes skills in isolated environments with resource limits
5. **REST API**: Exposes skills via FastAPI endpoints for easy integration

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      Skill Engine                            │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │   Discovery  │  │   Registry   │  │   Sandbox    │      │
│  │              │  │              │  │              │      │
│  │ • Scan dir   │  │ • Store meta │  │ • Execute    │      │
│  │ • Parse YAML │  │ • Index      │  │ • Isolate    │      │
│  │ • Watch      │  │ • Search     │  │ • Limit      │      │
│  └──────────────┘  └──────────────┘  └──────────────┘      │
│                                                               │
└─────────────────────────────────────────────────────────────┘
                        │
                        ▼
┌─────────────────────────────────────────────────────────────┐
│                    Skill Sources                             │
├─────────────────────────────────────────────────────────────┤
│  .claude/skills/                                             │
│  ├── quant-signal/         (Quant Skills)                    │
│  ├── quant-backtest/                                         │
│  ├── quant-risk/                                             │
│  ├── quant-decision/                                         │
│  ├── multi-model-debate/   (Agent Skills)                    │
│  ├── systematic-debugging/ (Debug Skills)                    │
│  └── ...                                                     │
└─────────────────────────────────────────────────────────────┘
```

## Skill Structure

Each skill is a directory under `.claude/skills/` with the following structure:

```
.claude/skills/
└── my-skill/
    ├── SKILL.md           # Required: Skill metadata and documentation
    ├── scripts/           # Optional: Executable scripts
    │   └── main.py        # Main entry point
    ├── prompts/           # Optional: LLM prompts
    ├── examples/          # Optional: Usage examples
    └── tests/             # Optional: Skill tests
```

### SKILL.md Format

```markdown
---
name: my-skill
description: Brief description of what this skill does
version: 1.0.0
author: Your Name
triggers:
  - "trigger phrase 1"
  - "trigger phrase 2"
dependencies:
  - package1
  - package2
type: quant  # quant, agent, debug, test, general
location: managed  # managed or user
---

# Skill Documentation

Detailed documentation here...

## When to Use

- Use case 1
- Use case 2

## Usage

Example code here...
```

## Python API

### Basic Usage

```python
from agentrt.skills import get_skill_engine

# Get global skill engine
engine = get_skill_engine()

# List all skills
all_skills = engine.list_skills()
print(f"Found {len(all_skills)} skills")

# List quant-specific skills
from agentrt.skills import SkillType
quant_skills = engine.list_skills(skill_type=SkillType.QUANT)

# Get skill metadata
skill = engine.get_skill("quant-signal")
print(f"Skill: {skill.name}")
print(f"Description: {skill.description}")
print(f"Triggers: {skill.triggers}")

# Search by trigger phrase
matching = engine.find_by_trigger("generate signal")
for skill in matching:
    print(f"Found: {skill.name}")
```

### Invoking Skills

#### Option 1: Python Handler (Recommended)

Register a Python function as a skill handler:

```python
from agentrt.skills import get_skill_engine

engine = get_skill_engine()

# Define handler
async def my_skill_handler(params: dict):
    symbol = params.get("symbol")
    timeframe = params.get("timeframe", "1d")

    # Your logic here
    result = {
        "symbol": symbol,
        "signal": "BUY",
        "confidence": 0.85
    }

    return result

# Register handler
engine.register_handler("my-skill", my_skill_handler)

# Invoke skill
result = await engine.invoke("my-skill", {
    "symbol": "AAPL",
    "timeframe": "1d"
})

print(f"Success: {result.success}")
print(f"Output: {result.output}")
print(f"Execution time: {result.execution_time_ms}ms")
```

#### Option 2: External Script

Execute a skill script in a sandboxed environment:

```python
# Skill will automatically execute scripts/main.py if no handler is registered
result = await engine.invoke("my-skill", {
    "symbol": "AAPL",
    "param1": "value1"
})

# Parameters are converted to command-line arguments:
# python scripts/main.py --symbol=AAPL --param1=value1
```

### Skill Statistics

```python
# Get engine statistics
stats = engine.get_stats()
print(f"Total skills: {stats['total_skills']}")
print(f"Skill types: {stats['skill_types']}")
print(f"Total triggers: {stats['total_triggers']}")
print(f"Registered handlers: {stats['registered_handlers']}")

# Get detailed skill info
info = engine.get_skill_info("quant-signal")
print(f"Has handler: {info['has_handler']}")
print(f"Script exists: {info['script_exists']}")
```

### Refreshing Skills

Refresh the skill registry after adding/modifying skills:

```python
count = engine.refresh_skills()
print(f"Loaded {count} skills")
```

## REST API

### List Skills

```bash
GET /agents/skills
GET /agents/skills?skill_type=quant
GET /agents/skills?location=managed
```

Response:
```json
[
  "quant-signal",
  "quant-backtest",
  "quant-risk",
  "quant-decision"
]
```

### Get Skill Info

```bash
GET /agents/skills/{skill_name}
```

Response:
```json
{
  "name": "quant-signal",
  "description": "Generate multi-strategy trading signals",
  "version": "1.0.0",
  "author": "TraderX Team",
  "triggers": ["generate signal", "trading signal"],
  "dependencies": ["pandas", "numpy"],
  "skill_type": "quant",
  "location": "managed",
  "has_handler": true,
  "script_exists": false
}
```

### Invoke Skill

```bash
POST /agents/skills/invoke
Content-Type: application/json

{
  "skill_name": "quant-signal",
  "params": {
    "symbol": "AAPL",
    "timeframe": "1d",
    "strategies": ["momentum", "mean_reversion"]
  },
  "timeout": 60
}
```

Response:
```json
{
  "skill_name": "quant-signal",
  "success": true,
  "output": {
    "symbol": "AAPL",
    "signal": "BUY",
    "confidence": 0.85,
    "strategies_used": ["momentum", "mean_reversion"]
  },
  "error": null,
  "execution_time_ms": 150,
  "executed_at": "2025-12-23T10:30:00Z"
}
```

### Search Skills

```bash
GET /agents/skills/search?query=signal
```

Response:
```json
{
  "query": "signal",
  "count": 2,
  "skills": [
    {
      "name": "quant-signal",
      "description": "Generate trading signals",
      "triggers": ["generate signal", "trading signal"],
      "skill_type": "quant"
    },
    {
      "name": "sentiment-signal",
      "description": "Sentiment-based signals",
      "triggers": ["sentiment signal"],
      "skill_type": "quant"
    }
  ]
}
```

### Refresh Skills

```bash
POST /agents/skills/refresh
```

Response:
```json
{
  "status": "success",
  "message": "Refreshed skill registry",
  "skills_loaded": 27
}
```

### Get Statistics

```bash
GET /agents/skills/stats
```

Response:
```json
{
  "total_skills": 27,
  "skill_types": {
    "quant": 4,
    "agent": 5,
    "debug": 3,
    "test": 2,
    "general": 13
  },
  "total_triggers": 85,
  "registered_handlers": 4,
  "skills_dir": "/path/to/.claude/skills"
}
```

## Creating Custom Skills

### Step 1: Create Skill Directory

```bash
mkdir .claude/skills/my-custom-skill
```

### Step 2: Create SKILL.md

```markdown
---
name: my-custom-skill
description: My custom skill for doing X
version: 1.0.0
author: Your Name
triggers:
  - "do custom thing"
  - "run my skill"
dependencies:
  - pandas
  - requests
type: general
location: user
---

# My Custom Skill

## Overview

This skill does X, Y, and Z.

## When to Use

Use this skill when you need to:
- Do X
- Process Y
- Analyze Z

## Usage

```python
result = await engine.invoke("my-custom-skill", {
    "param1": "value1"
})
```
```

### Step 3: Implement Handler (Option A)

Register a Python handler in your application:

```python
from agentrt.skills import get_skill_engine

engine = get_skill_engine()

async def my_custom_handler(params):
    # Implementation
    return {"result": "success"}

engine.register_handler("my-custom-skill", my_custom_handler)
```

### Step 4: Create Script (Option B)

Create `scripts/main.py`:

```python
#!/usr/bin/env python
"""
My Custom Skill Script
"""

import sys
import argparse
import json

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--param1", required=True)
    parser.add_argument("--param2", default="default")
    args = parser.parse_args()

    # Your logic here
    result = {
        "status": "success",
        "param1": args.param1,
        "param2": args.param2
    }

    # Output result as JSON
    print(json.dumps(result))

if __name__ == "__main__":
    main()
```

### Step 5: Test Your Skill

```python
from agentrt.skills import get_skill_engine

engine = get_skill_engine()
engine.refresh_skills()

result = await engine.invoke("my-custom-skill", {
    "param1": "test_value"
})

print(result.output)
```

## Skill Types

### Quant Skills
Financial analysis and trading skills:
- `quant-signal`: Trading signal generation
- `quant-backtest`: Strategy backtesting
- `quant-risk`: Risk analysis
- `quant-decision`: Trading decisions

### Agent Skills
Multi-agent coordination skills:
- `multi-model-debate`: Multi-model debates
- `dispatching-parallel-agents`: Parallel task execution
- `subagent-driven-development`: Subagent coordination

### Debug Skills
Debugging and troubleshooting:
- `systematic-debugging`: Systematic bug analysis
- `test-driven-development`: TDD workflows

### Test Skills
Testing utilities:
- `browser-automation-testing`: Browser automation
- `parallel-qa-orchestrator`: Parallel QA

### General Skills
General purpose utilities:
- `brainstorming`: Ideation
- `writing-plans`: Plan generation
- `verification-before-completion`: Verification checks

## Best Practices

### 1. Clear Naming
Use descriptive, hyphenated names:
- ✅ `quant-signal-generator`
- ❌ `qsg` or `QuantSignalGen`

### 2. Comprehensive Metadata
Provide complete YAML front matter:
- Clear description
- Relevant triggers
- Complete dependency list
- Appropriate skill type

### 3. Error Handling
Always handle errors gracefully:

```python
async def my_handler(params):
    try:
        # Your logic
        return {"status": "success"}
    except Exception as e:
        return {
            "status": "error",
            "message": str(e)
        }
```

### 4. Documentation
Include usage examples in SKILL.md:
- When to use
- Parameters
- Return format
- Error cases

### 5. Testing
Write tests for your skills:

```python
@pytest.mark.asyncio
async def test_my_skill():
    engine = get_skill_engine()
    result = await engine.invoke("my-skill", {"param": "value"})
    assert result.success
    assert result.output["status"] == "success"
```

### 6. Timeout Management
Set appropriate timeouts:
- Quick queries: 5-10 seconds
- Analysis tasks: 30-60 seconds
- Heavy computation: 5+ minutes

```python
result = await engine.invoke("heavy-skill", params, timeout=300)
```

## Security Considerations

### Sandboxing
Scripts are executed in sandboxed environments with:
- Limited environment variables
- Timeout enforcement
- Resource constraints

### Input Validation
Always validate skill parameters:

```python
async def my_handler(params):
    # Validate required params
    if "symbol" not in params:
        raise ValueError("symbol parameter required")

    # Sanitize inputs
    symbol = params["symbol"].upper()
    if not symbol.isalpha():
        raise ValueError("Invalid symbol format")
```

### Signature Verification
Future enhancement: Skills can be signed for verification:

```yaml
---
name: my-skill
signature: sha256:abc123...
---
```

## Troubleshooting

### Skill Not Found

```python
result = await engine.invoke("my-skill", {})
# Error: Skill not found: my-skill
```

**Solution**: Check skill name and refresh registry:
```python
engine.refresh_skills()
```

### Script Not Found

```python
# Error: Skill script not found at /path/to/scripts/main.py
```

**Solution**: Either:
1. Create `scripts/main.py` in skill directory
2. Register a Python handler

### Timeout Errors

```python
# Error: Skill execution timeout after 60s
```

**Solution**: Increase timeout:
```python
result = await engine.invoke("slow-skill", params, timeout=300)
```

### Dependency Issues

```python
# Error: ModuleNotFoundError: No module named 'pandas'
```

**Solution**: Install dependencies:
```bash
pip install pandas
```

## Performance Tips

### 1. Handler vs Script
- **Handler**: Faster, no subprocess overhead
- **Script**: Isolated, independent process

Use handlers for frequently called skills.

### 2. Caching
Cache expensive computations:

```python
from functools import lru_cache

@lru_cache(maxsize=100)
def expensive_calculation(param):
    # Heavy computation
    return result
```

### 3. Async Operations
Use async/await for I/O operations:

```python
async def my_handler(params):
    # Parallel API calls
    results = await asyncio.gather(
        fetch_data(symbol1),
        fetch_data(symbol2)
    )
    return results
```

### 4. Parallel Execution
Execute independent skills in parallel:

```python
from agentrt import AgentOrchestrator

tasks = [
    AgentTask(agent_type="skill", payload={"skill": "skill1"}),
    AgentTask(agent_type="skill", payload={"skill": "skill2"})
]

results = await orchestrator.dispatch_parallel(tasks)
```

## Integration with Agent Framework

Skills integrate seamlessly with the agent framework:

```python
from agentrt import AgentOrchestrator, AgentTask

orchestrator = get_orchestrator()

# Create skill task
task = AgentTask(
    agent_type="skill",
    payload={
        "skill_name": "quant-signal",
        "params": {"symbol": "AAPL"}
    }
)

# Dispatch through orchestrator
result = await orchestrator.dispatch(task)
```

## Future Enhancements

1. **Hot Reloading**: Auto-reload skills on file changes
2. **Versioning**: Multiple skill versions side-by-side
3. **Marketplace**: Skill sharing and discovery
4. **Metrics**: Execution tracking and performance monitoring
5. **Dependencies**: Automatic dependency installation
6. **Composition**: Chain skills together
7. **Scheduling**: Schedule skill execution

## References

- UQAP Architecture: `docs/architecture/UQAP-Universal-Quant-Agent-Platform.md`
- Example Skills: `.claude/skills/`
- Agent Framework: `backend-python/src/agents/`
- API Routes: `backend-python/src/api/agent_routes.py`
