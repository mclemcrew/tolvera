---
title: GSOC 2025
hide:
  - navigation
---

# GSOC 2025

Tölvera is joining Python's Google Summer of Code initiative for 2025.

> [Google Summer of Code (GSoC)](https://summerofcode.withgoogle.com/) is a global program that offers new contributors over 18 an *opportunity to be paid* for contributing to an open source project over a three month period.

Read on to find out how to apply.
Learn more about Python's GSOC initiatives [at their website](https://python-gsoc.org/).

## About Tölvera

Tölvera is a Python library for composing together and interacting with self-organising systems and artificial life. 
It provides creative coding-style APIs that allow users to combine and compose various built-in behaviours, such as flocking, slime mold growth, and swarming, and also author their own. 
With built-in support for Open Sound Control (OSC) and interactive machine learning (IML), Tölvera interfaces with and rapidly maps onto existing music software and hardware, striving to be both an accessible and powerful tool for exploring diverse intelligence in artistic contexts.

Tölvera is developed by [Dr. Jack Armitage](http://jackarmitage.com) and was originally created  at the [Intelligent Instruments Lab](http://iil.is), University of Iceland.
In 2024, Tölvera was awarded a grant from [Mozilla Builders](https://builders.mozilla.org/project/tolvera/).
This excerpt from an opinion piece for Mozilla outlines the motivation behind Tölvera:

!!! quote
    Picture the space of all possible forms of intelligence. Now zoom in — way in — until you see a tiny mote. That’s where we’re stuck right now: trapped in the speck of machine learning and large language models.

    While these tools are impressive, they represent just a fraction of this possibility space. AI discourse risks becoming a closed loop, with concerns about ML’s impact met only with proposals for better ML models, reinforcing our fixation. How do we build outward to see and benefit from intelligence in all its wonderfully diverse embodiments?
    
    – Jack Armitage, [Building Our Way Out – Beyond the Machine Learning Mote](https://builders.mozilla.org/building-our-way-out/), Mozilla Builders, December 2024

### Getting Started with Tölvera

- Install: `pip install tolvera` 
- [Documentation](https://tolvera.is)
- [Examples](https://github.com/Intelligent-Instruments-Lab/iil-examples/tree/main/tolvera)
- [Source Code](https://github.com/afhverjuekki/tolvera)
- [Discord Community](https://discord.gg/ER7tWds9vM)
- Read more about Tölvera's motivations and artistic usage in [publications](publications.md).

## Application Process

1. Join our [Discord Community](https://discord.gg/ER7tWds9vM) and introduce yourself.
2. Look through existing [examples](https://github.com/Intelligent-Instruments-Lab/iil-examples/tree/main/tolvera) and make one of your own. Alternatively, find and fix an issue and submit a pull request.
3.  Share your results on our Discord.
4. Discuss potential project ideas with mentors on Discord. Below you will find some project ideas, equally you are welcome to share your own idea.
5. Write a project application following the [PSF template](https://github.com/python-gsoc/python-gsoc.github.io/blob/master/ApplicationTemplate.md).
6. Share drafts of your application on Discord for discussion and feedback (this step is essential and valuable!).
7. Submit your application.

!!! warning
    We follow [Python GSOC's deadlines](https://python-gsoc.org/deadlines.html), which are separate to the official [GSOC Timeline](https://developers.google.com/open-source/gsoc/timeline).

## Project Mentors

The primary project mentor will be [Dr. Jack Armitage](http://jackarmitage.com), creator of Tölvera.
Jack has experience supervising GSOC projects with BeagleBoard.org and Bela.io, you can read the final publications here:

- [Differentiable Logic for Interactive Systems and Generative Music](https://ijc8.me/2024/08/26/gsoc-difflogic/)
- [Bela-IREE: An Approach to Embedded Machine Learning for Real-Time Music Interaction](https://aimc2023.pubpub.org/pub/t2l10z49/release/3) 

The secondary mentor(s) will be chosen from Tölvera contributors and collaborators, based on the project itself:

- [Victor Shepardson](https://iil.is/people#victor-shepardson), PhD Student, Intelligent Instruments Lab. Victor created `iipyper` and `anguilla`, the libraries that Tölvera relies on for OSC and IML.
- [Miguel Crozzoli](https://iil.is/people#miguel-angel-crozzoli), PhD Student, Intelligent Instruments Lab. Miguel is using Tölvera for sonification of climate datasets.
- [Piotr Rybicki](https://www.linkedin.com/in/piotr-rybicki-584940142/), Computer Scientist. Piotr has a keen interest in artificial life.
- [Robin Morabito](https://linktr.ee/bobhermit), Biologist & Artist. Robin has been experimenting with Tölvera for DNA visualisation.

## Project Ideas

### 1. Expand Artificial Life Model Library

- **Difficulty**: Intermediate
- **Size**: 350 hours (large)

**Description**: Expand Tölvera's library of basal behaviors and models (`tv.v`) by implementing new systems inspired by natural phenomena. Models could include ant colony optimization, predator-prey dynamics, chemotaxis, plant growth, cellular automata variants, and other complex adaptive systems. Each implementation should prioritize real-time performance and composability with existing models.

**Expected Outcomes**:

- At least 3 new working basal models with tests
- Documentation for each model explaining scientific background and parameters
- Example programs showing composition with existing models
- Performance benchmarks demonstrating real-time capability

**Required Skills**:

- Python programming
- Basic understanding of complex systems/artificial life
- Familiarity with numerical methods
- Interest in biological/physical systems

### 2. Generative AI/LLM Interface

- **Difficulty**: Intermediate
- **Size**: 350 hours (large)

**Description**: Create a new module (`tv.llm`) that enables natural language interaction with Tölvera. This includes developing a structured JSON representation of Tölvera programs for LLM manipulation, implementing prompt engineering for program generation/modification, and building an interactive CLI/UI for natural language control.

**Expected Outcomes**:

- JSON schema for Tölvera program representation
- Prompt engineering system for program manipulation
- CLI tool for natural language interaction
- Documentation of prompt design patterns
- Example programs showing common interaction patterns
- Test suite for LLM interactions

**Required Skills**:

- Python programming
- Experience with LLM APIs and prompt engineering
- Knowledge of JSON schemas and validation
- UI/UX design fundamentals

### 3. Physics Module

- **Difficulty**: Advanced
- **Size**: 350 hours (large)

**Description**: Design and implement a new physics module (`tv.phy`) that adds collision detection, fluid dynamics, and soft-body physics capabilities that can be composed with existing Tölvera models. The physics implementations should prioritize real-time performance and artistic exploration over physical accuracy, while maintaining believability.

**Expected Outcomes**:

- Modular physics engine supporting particles and basic shapes
- Integration with particle system (`tv.p`) and species system
- At least 3 example physics behaviors (collisions, fluids, soft-bodies)
- Documentation and tests
- Example programs demonstrating physics composition with other models

**Required Skills**:

- Strong Python programming skills
- Computer graphics and physics simulation experience
- GPU programming knowledge (Taichi preferred)
- Math background (linear algebra, numerical methods)

### 4. Scalable Particle System

- **Difficulty**: Intermediate/Advanced
- **Size**: 350 hours (large)

**Description**: Redesign Tölvera's particle system (`tv.p`) to handle millions of particles efficiently. This involves implementing spatial partitioning, GPU optimization, and multispecies interaction improvements. The goal is to enable much more complex scenes while maintaining real-time performance.

**Expected Outcomes**:

- Redesigned particle system with spatial acceleration
- Improved species interaction system
- Comprehensive benchmarking suite
- Migration guide for existing code
- Example programs demonstrating scale improvements
- Performance comparison across platforms

**Required Skills**:

- Strong Python and GPU programming
- Experience with particle systems
- Optimization and profiling skills
- Knowledge of spatial data structures

### 5. High Performance Computer Vision

- **Difficulty**: Intermediate/Advanced
- **Size**: 350 hours (large)

**Description**: Optimize Tölvera's computer vision module (`tv.cv`) to achieve reliable real-time performance with multiple tracking features enabled (hands, face, pose). This involves profiling and improving the integration of OpenCV and MediaPipe, implementing frame buffering and GPU acceleration where possible, and developing a robust camera input system.

**Expected Outcomes**:

- Redesigned video capture pipeline with improved performance
- GPU-accelerated preprocessing where possible
- Robust camera input system with error handling
- Comprehensive benchmarking suite
- Example programs demonstrating sustained real-time performance
- Cross-platform testing and optimization

**Required Skills**:

- Strong Python programming
- Experience with OpenCV and MediaPipe
- Video processing and real-time systems expertise
- GPU programming knowledge
- Performance optimization skills

### 6. Real-time OSC Mapping Engine

- **Difficulty**: Intermediate
- **Size**: 175 hours  (medium)

**Description**: Improve Tölvera's Open Sound Control (`tv.osc`) implementation by redesigning the mapping engine for better maintainability and extensibility. This includes streamlining the API, improving client integrations, and adding new features for music software interoperability.

**Expected Outcomes**:

- Redesigned OSC mapping API 
- Improved client generators for Max/MSP, PureData, SuperCollider
- Built-in OSC monitoring and debugging tools
- Migration guide for existing code
- Example programs showing common mapping patterns
- Performance benchmarks for real-time audio rate control

**Required Skills**:

- Python programming
- Knowledge of OSC protocol
- Familiarity with music software
- Real-time systems experience

## Contact

- Discord: [Join our community](https://discord.gg/ER7tWds9vM)
- Email: [Jack Armitage](mailto:ja@æ.is)
