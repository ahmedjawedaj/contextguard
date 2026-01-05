name: Bug report
description: Create a report to help us improve
title: "[BUG] "
labels: ["bug"]
body:
  - type: textarea
    id: summary
    attributes:
      label: Summary
      description: Clear, concise description of the bug.
    validations:
      required: true
  - type: textarea
    id: steps
    attributes:
      label: Steps to reproduce
      description: Include commands, inputs, and expected vs actual behavior.
      placeholder: |
        1. ...
        2. ...
    validations:
      required: true
  - type: textarea
    id: context
    attributes:
      label: Environment
      description: OS, Python version, optional deps installed (llm/qdrant/chroma/cloud), relevant settings.
  - type: textarea
    id: logs
    attributes:
      label: Logs / tracebacks
      description: Paste relevant logs (redact secrets).
  - type: checkboxes
    id: checklist
    attributes:
      label: Checklist
      options:
        - label: I have searched existing issues.
        - label: I can reproduce this on the latest main.

