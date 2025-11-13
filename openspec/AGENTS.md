# OpenSpec Agent Workflow

This document explains how AI assistants should work with OpenSpec in this project.

## What is OpenSpec?

OpenSpec is a structured workflow for managing changes in AI-assisted development. It ensures:
- All significant changes are documented before implementation
- Clear approval gates before coding begins
- Maintainable history of decisions and implementations
- Reduced rework and miscommunication

## The OpenSpec Workflow

### 1. Proposal Phase (Planning)

**When to create a proposal:**
- New features or capabilities
- Breaking changes
- Architecture changes
- Significant refactoring
- Performance optimizations
- Security enhancements
- Any change that needs discussion or approval

**How to create a proposal:**

Use the `/openspec:proposal` slash command or follow these steps:

1. Create a new file in `openspec/changes/` named `NNNN-brief-description.md`
   - NNNN is a 4-digit sequential number (0001, 0002, etc.)
   - Use lowercase with hyphens for the description

2. Use this template:

```markdown
# [Brief Title]

**Status:** proposal
**Created:** YYYY-MM-DD
**Updated:** YYYY-MM-DD

## Summary
[1-2 sentences explaining what and why]

## Motivation
[Why is this change needed? What problem does it solve?]

## Proposed Changes
[Detailed description of the changes]

### Files to Create
- `path/to/new/file.py` - Description

### Files to Modify
- `path/to/existing/file.py` - What will change

### Files to Delete
- `path/to/old/file.py` - Why it's being removed

## Implementation Plan
1. [Step 1]
2. [Step 2]
3. [Step 3]

## Testing Strategy
[How will this be tested?]

## Risks and Considerations
[Potential issues, breaking changes, migration needs]

## Alternatives Considered
[Other approaches that were considered and why they weren't chosen]
```

3. Present the proposal to the user for review and approval

**Proposal validation:**
- Check that the spec follows the template structure
- Ensure all sections are filled out thoughtfully
- Verify file paths are accurate
- Confirm the implementation plan is detailed and actionable

### 2. Approval Phase

**User reviews the proposal and:**
- Approves → Change status to `approved`, proceed to Apply phase
- Requests changes → Update the proposal based on feedback
- Rejects → Archive with status `rejected`

**Important:** NEVER begin implementation until status is `approved`

### 3. Apply Phase (Implementation)

**Use the `/openspec:apply` slash command** with the change number:

```
/openspec:apply NNNN
```

This will:
1. Load the approved spec
2. Guide you through implementation
3. Keep the spec updated with progress
4. Track tasks and ensure completeness

**During implementation:**
- Follow the spec exactly as approved
- Update the spec's status to `in-progress`
- Use TodoWrite to track implementation tasks
- If you discover issues, pause and update the proposal
- Test thoroughly according to the testing strategy

**When implementation is complete:**
- Update status to `implemented`
- Verify all tasks are done
- Run all tests
- Document any deviations from the original plan

### 4. Archive Phase

**Use the `/openspec:archive` slash command:**

```
/openspec:archive NNNN
```

This will:
1. Move the spec to `openspec/archive/`
2. Update status to `archived`
3. Add completion date
4. Update project documentation if needed

## Key Principles

### 1. Always Spec First
- For significant changes, ALWAYS create a proposal before coding
- The spec is the source of truth
- Code should match the approved spec

### 2. Incremental Changes
- Break large features into smaller, manageable changes
- Each change should be focused and well-scoped
- Related changes can reference each other

### 3. Clear Communication
- Proposals should be clear and detailed
- Use the spec to align understanding
- Ask for clarification if requirements are ambiguous

### 4. Maintain History
- Keep all specs (even rejected ones) for context
- Archive implemented specs for reference
- Document deviations and why they occurred

### 5. Test Coverage
- Every change should include a testing strategy
- Tests should be implemented alongside features
- Document how to verify the change works

## Working with the User

### When the user asks for a feature:

1. **Assess complexity:**
   - Simple/small change → May proceed directly
   - Complex/significant change → Create proposal first

2. **Create proposal if needed:**
   - Use `/openspec:proposal` or create manually
   - Present to user for approval
   - Wait for explicit approval

3. **Implement when approved:**
   - Use `/openspec:apply NNNN`
   - Follow the spec strictly
   - Update status as you progress

4. **Archive when complete:**
   - Use `/openspec:archive NNNN`
   - Ensure all documentation is updated

### When requirements are unclear:

1. **Ask questions** before creating the proposal
2. **Draft the proposal** with your understanding
3. **Present it to the user** to confirm alignment
4. **Iterate** until approved

### When implementation deviates from spec:

1. **Stop implementation**
2. **Update the proposal** with new understanding
3. **Get user approval** for the changes
4. **Continue** with updated plan

## File Organization

```
openspec/
├── AGENTS.md          # This file - workflow for AI agents
├── project.md         # Project context, tech stack, conventions
├── changes/           # Active proposals and specs
│   ├── 0001-feature-name.md
│   └── 0002-another-feature.md
└── archive/           # Completed or rejected specs
    ├── 0001-feature-name.md
    └── 0002-another-feature.md
```

## Slash Commands

The project includes three slash commands for OpenSpec:

- `/openspec:proposal` - Create a new change proposal
- `/openspec:apply NNNN` - Implement an approved change
- `/openspec:archive NNNN` - Archive a completed change

Use these commands to streamline the workflow.

## Best Practices

1. **Be thorough in proposals** - Better to over-document than under-document
2. **Keep specs updated** - The spec should reflect current understanding
3. **Use TodoWrite** - Track implementation tasks clearly
4. **Test rigorously** - Follow the testing strategy in the spec
5. **Communicate clearly** - Use the spec to align with the user
6. **Archive promptly** - Keep the changes/ directory focused on active work
7. **Reference project.md** - Follow established conventions and patterns

## Example Workflow

```
User: "I want to add a caching layer to improve performance"

Agent: "I'll create an OpenSpec proposal for adding a caching layer.
        This is a significant architectural change that should be
        planned carefully."

[Agent creates openspec/changes/0003-caching-layer.md]

Agent: "I've created a proposal in openspec/changes/0003-caching-layer.md.
        Please review the approach, implementation plan, and testing
        strategy. Let me know if you'd like any changes or if you
        approve this plan."

User: "Looks good, but use Redis instead of in-memory cache"

[Agent updates the proposal]

Agent: "I've updated the proposal to use Redis. Ready for approval?"

User: "Approved"

Agent: "I'll begin implementation using /openspec:apply 0003"

[Agent implements following the spec, using TodoWrite to track tasks]

Agent: "Implementation complete. All tests passing. Would you like me
        to archive this change?"

User: "Yes"

[Agent uses /openspec:archive 0003]

Agent: "Change 0003 has been archived. The caching layer is now live."
```

## Questions?

If you're unsure about any aspect of the OpenSpec workflow:
1. Refer to this document
2. Check project.md for project-specific context
3. Ask the user for clarification
4. Default to creating a proposal when in doubt
