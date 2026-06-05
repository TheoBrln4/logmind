## Overview

LogMind is an AI-powered Root Cause Analysis (RCA) platform designed to help engineers investigate incidents faster.

The system combines agentic workflows, Retrieval-Augmented Generation (RAG), vector search and large language models to automatically analyze logs and generate incident reports.

Instead of manually inspecting thousands of log lines, LogMind identifies patterns, retrieves similar historical incidents and proposes likely root causes.

## How It Works

LogMind processes logs through a multi-agent workflow built with LangGraph.

### 1. Log Parsing

Raw logs are transformed into structured events containing:

- timestamp
- log level
- service name
- message

This standardization allows downstream agents to work with consistent data.

### 2. Pattern Detection

Logs are embedded and clustered using DBSCAN.

This step helps identify recurring error patterns and group similar events together.

### 3. Historical Incident Retrieval

Relevant incidents are retrieved from ChromaDB using semantic similarity search.

This provides additional context based on previously observed failures.

### 4. Root Cause Analysis

The RCA Agent combines:

- current log patterns
- historical incidents
- contextual information

to generate hypotheses regarding the most likely root cause.

### 5. Report Generation

A final report is generated using Qwen and contains:

- root cause hypotheses
- supporting evidence
- impact analysis
- remediation recommendations

### 6. Incident Memory

The analyzed incident is stored in ChromaDB and can later be reused as context for future investigations.

## Architecture

```mermaid
graph TD

A[Raw Logs]
--> B[Parser Agent]

B --> C[Embedding Agent]

C --> D[Pattern Agent]

D --> E[RCA Agent]

E --> F[Report Agent]

F --> G[Persist Agent]

G --> H[(ChromaDB)]

H --> E

E --> I[Qwen]


---

## Installation

Comme tu utilises Docker Desktop :

```md
## Installation

### Prerequisites

- Docker Desktop
- Git

### Clone the repository

```bash
git clone https://github.com/TheoBrln4/logmind.git
cd logmind

docker compose up --build
