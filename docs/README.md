# AIME Documentation

**Adaptive Infinite Mixture Engine** - A world model for embodied AI grounded in active inference

---

## 📚 Documentation Index

### 🎯 Start Here

| Document | Purpose | Read This If... |
|----------|---------|-----------------|
| **[AI_CODER_QUICKSTART.md](AI_CODER_QUICKSTART.md)** | 5-minute onboarding for AI assistants | You're an AI helping with this codebase for the first time |
| **[NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md)** | "I want to understand X" → "Read file Y" | You need to find specific code quickly |

### 🧠 Understanding AIME

| Document | Purpose | Read This If... |
|----------|---------|-----------------|
| **[THEORY_AND_PHILOSOPHY.md](THEORY_AND_PHILOSOPHY.md)** | Theoretical foundations and vision | You want to understand WHY AIME is designed this way |
| **[ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md)** | System diagrams and component interactions | You want to see HOW components fit together |
| **[TENSOR_SHAPE_REFERENCE.md](TENSOR_SHAPE_REFERENCE.md)** | Detailed tensor shape flows | You're debugging shape mismatches or tracing data flow |

### 🔧 Development

| Document | Purpose | Read This If... |
|----------|---------|-----------------|
| **[REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md)** | Refactoring roadmap and rationale | You're helping reorganize the codebase |
| **[CONTRIBUTING.md](CONTRIBUTING.md)** | How to add new features or fix bugs | You want to contribute code *(TODO)* |

---

## 🚀 Quick Navigation by Goal

### "I'm completely new to this codebase"
1. Read: [AI_CODER_QUICKSTART.md](AI_CODER_QUICKSTART.md) (5 min)
2. Read: [THEORY_AND_PHILOSOPHY.md](THEORY_AND_PHILOSOPHY.md) - Executive Summary section (10 min)
3. Skim: [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) - Look at diagrams (5 min)
4. Use: [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md) - As reference when diving into code

**Total time to productivity: ~20 minutes**

---

### "I need to debug a specific issue"
1. Identify component: [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md) - Task-Based Navigation
2. Understand shapes: [TENSOR_SHAPE_REFERENCE.md](TENSOR_SHAPE_REFERENCE.md) - Find relevant section
3. Read theory: [THEORY_AND_PHILOSOPHY.md](THEORY_AND_PHILOSOPHY.md) - Find component section
4. Trace code: Use file locations from [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md)

---

### "I want to add a new feature"
1. Understand design principles: [THEORY_AND_PHILOSOPHY.md](THEORY_AND_PHILOSOPHY.md) - Design Principles section
2. See where it fits: [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) - Component diagram
3. Find similar code: [NAVIGATION_GUIDE.md](NAVIGATION_GUIDE.md) - Class locations table
4. Check refactoring plan: [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md) - See if it affects structure

---

### "I'm helping refactor the codebase"
1. Understand WHY: [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md) - Theory-Driven Refactoring section
2. Understand WHAT: [THEORY_AND_PHILOSOPHY.md](THEORY_AND_PHILOSOPHY.md) - Five Pillars section
3. See WHERE to move files: [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md) - Target Directory Structure
4. Track progress: [REORGANIZATION_PLAN.md](REORGANIZATION_PLAN.md) - Session checklists

---

## 📖 Document Descriptions

### AI_CODER_QUICKSTART.md
**The 5-minute onboarding guide**

Essential information for AI coding assistants:
- What is AIME? (1 paragraph)
- Where is the code? (directory structure)
- How do I run it? (quick commands)
- Where do I go next? (decision tree)

Target audience: AI assistants encountering this codebase for the first time

---

### NAVIGATION_GUIDE.md
**The "find things fast" reference**

Maps tasks to files:
- "I want to understand attention" → Read these 4 files
- "Where is DPGMMPrior?" → Line 379 of this file
- Class location table with current + future locations
- Dependency graph
- AI coder workflow examples

Target audience: Anyone who knows what they want to find but not where it is

---

### THEORY_AND_PHILOSOPHY.md
**The "why does AIME exist" document**

Comprehensive theoretical foundation:
- Core philosophy (adaptive infinite mixtures)
- Connection to active inference
- Five Pillars (Perception, Representation, Dynamics, Attention, Optimization)
- Mathematical framework (ELBO, DPGMM, stick-breaking)
- Design principles
- Future vision

Target audience: Researchers, collaborators, future developers understanding the vision

---

### ARCHITECTURE_OVERVIEW.md
**The "how does it work" visual guide**

System diagrams and component interactions:
- High-level architecture (ASCII diagrams)
- Data flow through the model
- Component dependency graph
- Module interaction patterns
- Training loop visualization

Target audience: Visual learners, system designers, integration engineers

---

### TENSOR_SHAPE_REFERENCE.md
**The "what are the shapes" debugging aid**

Complete tensor transformations:
- Shape flow through full forward pass
- Per-component shape documentation
- Common shape errors and solutions
- Concrete examples with real dimensions
- Shape validation checklist

Target audience: Debuggers, feature developers, integration testing

---

### REORGANIZATION_PLAN.md
**The "how to restructure" roadmap**

Refactoring strategy and plan:
- Current pain points analysis
- Theory-driven reorganization approach
- Target directory structure (5 Pillars)
- 7-session implementation plan
- Expected benefits
- Migration checklist

Target audience: Refactoring contributors, project managers, AI coding teams

---

## 🎨 Documentation Principles

These docs follow specific design principles:

### 1. Layered Complexity
- **Quick Start** (5 min) → **Navigation** (as-needed) → **Theory** (deep dive)
- Start shallow, go deeper as needed

### 2. Multiple Entry Points
- By goal ("I want to debug")
- By role ("I'm a researcher")
- By component ("How does attention work?")

### 3. AI-Coder Optimized
- Clear file:line references
- Self-contained sections (minimize jumping)
- Copy-pasteable code examples
- Explicit shape annotations

### 4. Theory-Grounded
- Every code component maps to cognitive function
- Design decisions explained, not just described
- Connect implementation to neuroscience/ML literature

### 5. Living Documents
- Updated as code evolves
- Version controlled alongside code
- Cross-references stay synchronized

---

## 📊 Documentation Status

| Document | Status | Last Updated | Completeness |
|----------|--------|--------------|--------------|
| README.md | ✅ Complete | 2025-11-01 | 100% |
| AI_CODER_QUICKSTART.md | 🚧 In Progress | - | 0% |
| NAVIGATION_GUIDE.md | ✅ Complete | 2025-11-01 | 100% |
| THEORY_AND_PHILOSOPHY.md | ✅ Complete | 2025-11-01 | 100% |
| ARCHITECTURE_OVERVIEW.md | 🚧 In Progress | - | 0% |
| TENSOR_SHAPE_REFERENCE.md | 🚧 In Progress | - | 0% |
| REORGANIZATION_PLAN.md | ✅ Complete | 2025-11-01 | 100% |
| CONTRIBUTING.md | ⏳ Planned | - | 0% |

Legend: ✅ Complete | 🚧 In Progress | ⏳ Planned | ❌ Deprecated

---

## 🤝 Contributing to Documentation

When updating docs:

1. **Keep cross-references synced**: If you move code, update NAVIGATION_GUIDE.md
2. **Update status table**: Mark documents as you edit them
3. **Maintain theory-code alignment**: Code changes may require THEORY_AND_PHILOSOPHY.md updates
4. **Add examples**: Concrete examples > abstract descriptions
5. **Test AI comprehension**: Can an AI assistant follow your instructions?

---

## 📝 Feedback

Found something unclear? Have suggestions?
- Open an issue or PR
- Update the doc directly (it's version controlled)
- Ask a human or AI collaborator

---

*"Documentation is love letters to your future self (and future AI assistants)."*
