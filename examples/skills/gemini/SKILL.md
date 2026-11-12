# Gemini CLI Integration Skill

## Purpose
Enable seamless integration with Google's Gemini CLI for AI-powered code analysis, generation, and complex reasoning tasks within the development workflow.

## When to Use This Skill
- When you need to leverage Gemini's reasoning capabilities
- When performing documentation generation
- When you want a different AI perspective on complex problems
- When working with multimodal content (code + images/diagrams)

## Context
Gemini CLI is Google's command-line interface for Gemini models. It excels at complex reasoning, documentation generation, and can handle multimodal inputs.

## Methodology

### 1. Task Identification
Identify tasks suitable for Gemini:
- Complex reasoning and problem-solving
- Documentation generation
- Code explanation and teaching
- Architecture design discussions
- Multimodal analysis (code + diagrams)

### 2. Invocation Pattern
```bash
# Basic usage
gemini "Your prompt here"

# With file context
gemini -f src/file.ts "Explain this code in detail"

# With sandbox mode for code execution
gemini -s "Generate and test a sorting algorithm"
```

### 3. Configuration
Use GEMINI.md file for persistent configuration:
```markdown
# GEMINI.md
- Use TypeScript for all code generation
- Follow functional programming patterns
- Include comprehensive documentation
```

### 4. Output Handling
- Review Gemini's suggestions for accuracy
- Validate generated documentation
- Cross-reference with existing code patterns
- Apply consistent formatting

## Anti-Patterns
- **Context mismatch**: Don't use for tasks requiring deep project context
- **Over-reliance**: Balance with local analysis and reasoning
- **Ignoring limitations**: Be aware of Gemini's knowledge cutoff
- **Unstructured prompts**: Provide clear, structured instructions

## Examples

### Documentation Generation
```bash
gemini -f src/api/routes.ts "Generate OpenAPI documentation for these routes"
```

### Architecture Review
```bash
gemini -f src/services/*.ts "Review the service layer architecture and suggest improvements"
```

### Code Explanation
```bash
gemini -f src/algorithms/sort.ts "Explain this algorithm step by step for a junior developer"
```

## Verification
After using Gemini:
1. Verify generated documentation accuracy
2. Cross-check suggestions with project requirements
3. Ensure consistency with existing patterns
4. Review for completeness
