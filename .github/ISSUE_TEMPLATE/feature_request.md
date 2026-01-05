name: Feature request
description: Suggest an idea for this project
title: "[FEATURE] "
labels: ["enhancement", "good first issue"]
body:
  - type: textarea
    id: summary
    attributes:
      label: Summary
      description: What problem are you trying to solve? Describe the feature clearly.
    validations:
      required: true
  - type: textarea
    id: proposal
    attributes:
      label: Proposal
      description: How should it work? Include API/UX ideas, constraints, acceptance criteria.
    validations:
      required: true
  - type: textarea
    id: alternatives
    attributes:
      label: Alternatives considered
      description: List any alternative solutions or workarounds you’ve considered.
  - type: checkboxes
    id: checklist
    attributes:
      label: Checklist
      options:
        - label: I have searched existing issues.
        - label: This aligns with the project scope (constraint-first verification).

