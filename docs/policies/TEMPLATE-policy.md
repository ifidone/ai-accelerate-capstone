# [Your Domain] Policy Document — Template

## Overview

Write a short description of what this document covers. Your agent will
retrieve sections of this document to answer user questions — so write
it the way a real policy, guide, or knowledge base article would read.

---

## Section 1: [Topic]

Write the rules, guidelines, or information for this topic here.

If your domain has categories, segments, or types that have *different* rules,
put each one in its own section (see below). This is important for RAG — the
agent needs to retrieve the right section for the right user.

---

## Section 2: [Category A — e.g., a user type, region, or tier]

Write rules or information specific to this category.

Example: If you're building a restaurant recommender, this might be
"Vegetarian options" or "Budget-friendly picks."

---

## Section 3: [Category B]

Write rules or information specific to this category.

---

## What to keep in mind when writing your documents

- **Be specific.** Vague documents produce vague answers. Your agent can
  only be as precise as your source material.
- **Use consistent headings.** Your chunking strategy will rely on them.
- **One topic per section.** Don't mix unrelated rules in the same block.
- **Include edge cases.** "What happens if X?" is often what users ask.
- **Write for retrieval.** Short, dense sections with clear headers retrieve
  better than walls of text.

You should have at least 2–3 documents covering different aspects of your
domain. Copy this file, rename it, and fill it in.
