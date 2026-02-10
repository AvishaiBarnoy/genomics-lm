# Product Guidelines

## Communication & Documentation
- **Technical and Direct:** All communication, documentation, and code comments should prioritize technical precision. Use standard ML and genomics terminology (e.g., "stochastic gradient descent", "codon usage bias") directly, assuming a professional background from the target audience. The goal is efficiency and clarity for experts.

## Analysis & Tooling
- **CLI-First with Integrated Comparison:** Tools should prioritize clean, parsable terminal outputs (CSV/JSON/Logs) to facilitate rapid, reproducible experimentation. Additionally, emphasis should be placed on building integrated views and dashboards that allow for direct comparison of multiple runs across the entire 6-step interpretability pipeline.

## Development Principles
- **Modular Efficiency:** Development must balance strict modularity (clear, isolated interfaces for agents) with end-to-end performance. While agents should be independently testable and replaceable, the underlying implementation should leverage shared state or global optimizations where they provide significant gains in training speed or resource efficiency on consumer hardware.
