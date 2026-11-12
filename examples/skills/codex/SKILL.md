# Codex CLI Integration Skill

## Purpose
Enable seamless integration with OpenAI's Codex CLI for code analysis, refactoring, and automated code changes within the development workflow.

## When to Use This Skill
- When you need to delegate complex code analysis tasks to Codex
- When performing large-scale refactoring operations
- When you want a second opinion on code quality or implementation
- When generating boilerplate code or repetitive patterns

## Context
Codex CLI is OpenAI's command-line interface for AI-powered code generation and analysis. It can be used alongside Claude Code to leverage different AI models for specific tasks.

## Methodology

### 1. Task Identification
Identify tasks suitable for Codex:
- Code generation from natural language
- Code completion and refactoring
- Bug detection and fixing suggestions
- Code explanation and documentation

### 2. Invocation Pattern
```bash
# Basic usage
codex "Your prompt here"

# With file context using @ syntax
codex @src/file.ts "Analyze this file for potential bugs"

# With multiple files
codex @src/main.ts @src/utils.ts "Refactor these to share common logic"
```

### 3. Output Handling
- Review Codex suggestions before applying
- Validate generated code against project standards
- Run tests after applying changes
- Document significant refactoring decisions

## Anti-Patterns
- **Blind acceptance**: Never apply Codex suggestions without review
- **Context overload**: Don't include too many files at once
- **Vague prompts**: Be specific about what you want
- **Ignoring project style**: Ensure generated code matches project conventions

## Examples

### Code Analysis
```bash
codex @src/components/Form.tsx "Review this component for accessibility issues"
```

### Refactoring
```bash
codex @src/utils/helpers.ts "Refactor this file to use TypeScript generics"
```

### Bug Investigation
```bash
codex @src/api/client.ts "Find potential race conditions in this code"
```

## Verification
After using Codex:
1. Review all suggested changes
2. Run the test suite
3. Check for regressions
4. Verify code style compliance