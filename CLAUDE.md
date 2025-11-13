<!-- OPENSPEC:START -->
# OpenSpec Instructions

These instructions are for AI assistants working in this project.

Always open [openspec/AGENTS.md](openspec/AGENTS.md) when the request:
- Mentions planning or proposals (words like proposal, spec, change, plan)
- Introduces new capabilities, breaking changes, architecture shifts, or big performance/security work
- Sounds ambiguous and you need the authoritative spec before coding

Use [openspec/AGENTS.md](openspec/AGENTS.md) to learn:
- How to create and apply change proposals
- Spec format and conventions
- Project structure and guidelines

Keep this managed block so 'openspec update' can refresh the instructions.

<!-- OPENSPEC:END -->

# Project-Specific Instructions

## Project Context
This is a Xiaohongshu (小红书) SEO note content optimization agent built with CrewAI framework.

For detailed project information, see [openspec/project.md](openspec/project.md).

## Working with This Project

1. **Always check [openspec/project.md](openspec/project.md)** for:
   - Tech stack and dependencies
   - Coding conventions
   - Project structure
   - Common patterns

2. **Use OpenSpec workflow for significant changes**:
   - New features → Create proposal first
   - Breaking changes → Create proposal first
   - Simple fixes → May proceed directly
   - When unsure → Create proposal

3. **Follow the conventions** documented in project.md:
   - Code style
   - Testing approach
   - Documentation standards
   - Error handling

## Getting Started

New AI assistants should:
1. Read [openspec/project.md](openspec/project.md) to understand the project
2. Read [openspec/AGENTS.md](openspec/AGENTS.md) to understand the workflow
3. Ask the user about any unclear conventions or requirements
4. Propose changes using OpenSpec for any significant work

## Slash Commands

This project includes three OpenSpec slash commands:
- `/openspec:proposal` - Create a new change proposal
- `/openspec:apply NNNN` - Implement an approved change
- `/openspec:archive NNNN` - Archive a completed change

Use these to streamline your workflow.
