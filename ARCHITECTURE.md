# Architecture & Flow Diagrams

This document contains visual diagrams of the system architecture and agent flows.

---

## 1. Core Perception-Decision-Action Loop

![Core PDA Loop](pda_loop.svg)

```mermaid
graph TD
    A["🌍 Environment<br/>User Input / External Event"] --> B["👁 PERCEIVE<br/>normalize, encode, clean"]
    B --> C["🧠 DECIDE<br/>choose action<br/>analyze, predict, reason"]
    C --> D["⚡ ACT<br/>execute action<br/>call API, return result"]
    D --> E["💾 MEMORY<br/>store in episodic memory<br/>for future context"]
    E --> F["📤 Return Result<br/>to caller"]
    
    style A fill:#e1f5ff
    style B fill:#fff3e0
    style C fill:#f3e5f5
    style D fill:#e8f5e9
    style E fill:#fce4ec
    style F fill:#f0f4c3
```

---

## 2. Agent Type Comparison

![Agent Types](agent_types.svg)

```mermaid
graph TB
    subgraph BaseAgent["🤖 BaseAgent<br/>(abstract interface)"]
        interface["perceive()<br/>decide()<br/>act()<br/>run()"]
    end
    
    subgraph Reactive["⚡ ReactiveAgent<br/>(Phase 1)"]
        r1["Memory: rule table<br/>Speed: ~1ms<br/>Learning: none"]
    end
    
    subgraph Learning["📊 LearningAgent<br/>(Phase 1)"]
        l1["Model: TF-IDF + LogReg<br/>Speed: ~5ms<br/>Training: sklearn.pipeline"]
    end
    
    subgraph LLM["🧠 LLMAgent<br/>(Phase 2)"]
        llm1["Providers: OpenAI, Anthropic, Ollama<br/>Speed: 300-2000ms<br/>Memory: sliding window history"]
    end
    
    subgraph ToolUsing["🔧 ToolUsingAgent<br/>(Phase 3)"]
        t1["Loop: reason → act → observe<br/>Tools: web_search, file_io, http<br/>Max iterations: configurable"]
    end
    
    subgraph Orchestrator["🎼 Orchestrator<br/>(Phase 4)"]
        orch1["Planner → Executor → Reviewer<br/>Inter-agent message queue<br/>Shared short-term memory"]
    end
    
    BaseAgent --> Reactive
    BaseAgent --> Learning
    BaseAgent --> LLM
    BaseAgent --> ToolUsing
    BaseAgent --> Orchestrator
    
    style BaseAgent fill:#e0e0e0
    style Reactive fill:#fff3e0
    style Learning fill:#e3f2fd
    style LLM fill:#f3e5f5
    style ToolUsing fill:#e8f5e9
    style Orchestrator fill:#fce4ec
```

---

## 3. ReactiveAgent Flow

![ReactiveAgent Flow](reactive_flow.svg)

```mermaid
sequenceDiagram
    participant User
    participant Agent as ReactiveAgent
    participant RuleTable as Rule Lookup
    participant Memory as Episodic Memory

    User->>Agent: "What is the price?"
    activate Agent
    
    Agent->>Agent: perceive()
    Note over Agent: normalize to lowercase<br/>"what is the price?"
    
    Agent->>Agent: decide()
    Agent->>RuleTable: lookup "price"
    RuleTable-->>Agent: "Our pricing page is at /pricing"
    
    Agent->>Agent: act()
    Note over Agent: return matched response
    
    Agent->>Memory: store(observation,<br/>action, result)
    
    Agent-->>User: "Our pricing page is at /pricing"
    deactivate Agent
```

---

## 4. LearningAgent Flow

![LearningAgent Flow](learning_flow.svg)

```mermaid
graph TD
    subgraph Training["🎓 TRAINING PHASE"]
        T1["add_examples()<br/>24 labelled pairs:<br/>text → intent"]
        T2["train()<br/>TF-IDF vectorize<br/>LogisticRegression.fit()"]
        T3["evaluate()<br/>train/test split<br/>classification_report"]
        T4["is_trained = True"]
        
        T1 --> T2 --> T3 --> T4
    end
    
    subgraph Inference["🔮 INFERENCE PHASE"]
        I1["User query"]
        I2["perceive()<br/>normalize text"]
        I3["decide()<br/>pipeline.predict_proba()"]
        I4["act()<br/>map intent → response"]
        I5["return result"]
        
        I1 --> I2 --> I3 --> I4 --> I5
    end
    
    Training -.->|when is_trained=True| Inference
    
    style Training fill:#e3f2fd
    style Inference fill:#f3e5f5
```

---

## 5. LLMAgent Flow

![LLMAgent Flow](llm_flow.svg)

```mermaid
graph TD
    subgraph Init["⚙️ INIT"]
        I1["env vars:<br/>AGENT_PROVIDER<br/>OPENAI_API_KEY"]
        I2["build client<br/>OpenAI()<br/>Anthropic()"]
        I1 --> I2
    end
    
    subgraph Loop["🔄 CONVERSATION LOOP"]
        L1["User message"]
        L2["perceive()<br/>normalize"]
        L3["append to history<br/>user_message"]
        L4["decide()<br/>build_messages<br/>system + history + user"]
        L5["act()<br/>LLM API call"]
        L6["extract response<br/>append to history"]
        L7["return to caller"]
        
        L1 --> L2 --> L3 --> L4 --> L5 --> L6 --> L7
    end
    
    Init -.-> Loop
    
    style Init fill:#f3e5f5
    style Loop fill:#e8f5e9
```

---

## 6. ToolUsingAgent Flow (Phase 3)

![ToolUsingAgent Flow](tool_using_flow.svg)

```mermaid
graph TD
    A["User Query<br/>e.g., 'Bitcoin price'"] 
    B["LLM Thinks<br/>decide() → 'call web_search'"]
    C["Tool Call<br/>act() → web_search(...)"]
    D["Tool Executes"]
    E["Observe Result"]
    F{"Done?"}
    G["Return Answer"]
    
    A --> B
    B --> C
    C --> D
    D --> E
    E --> F
    F -->|No, loop| B
    F -->|Yes| G
    
    style A fill:#e1f5ff
    style B fill:#f3e5f5
    style C fill:#e8f5e9
    style D fill:#fff3e0
    style E fill:#fce4ec
    style F fill:#f0f4c3
    style G fill:#c8e6c9
```

---

## 7. Multi-Agent Orchestration (Phase 4)

![Orchestrator Flow](orchestrator_flow.svg)

```mermaid
graph TD
    User["👤 User<br/>Complex Goal"]
    
    subgraph Planner["📋 PlannerAgent"]
        P1["Goal: 'Summarize all<br/>company policies'"]
        P2["→ Subtask 1: Find policy docs<br/>→ Subtask 2: Extract key points<br/>→ Subtask 3: Synthesize summary"]
    end
    
    subgraph Executor["⚙️ ExecutorAgent"]
        E1["Subtask 1: web_search<br/>→ 'https://policies.pdf'"]
        E2["Subtask 2: read_file<br/>→ extract key points"]
        E3["Subtask 3: LLM synthesize<br/>→ summary"]
    end
    
    subgraph Reviewer["✅ ReviewerAgent"]
        R1["Is summary complete?"]
        R2["Ask executor for missing info"]
        R3["Approve & return"]
    end
    
    Queue["📬 Message Queue<br/>shared memory"]
    
    User --> Planner
    Planner -->|subtasks| Queue
    Queue -->|poll| Executor
    Executor -->|results| Queue
    Queue -->|poll| Reviewer
    Reviewer -->|validate| Executor
    Reviewer -->|final answer| User
    
    style Planner fill:#e3f2fd
    style Executor fill:#e8f5e9
    style Reviewer fill:#fce4ec
    style Queue fill:#fff3e0
```

---

## 8. System Architecture (Full Stack)

![Full Stack Architecture](full_stack.svg)

```mermaid
graph TB
    subgraph Client["🖥️ CLIENT"]
        Web["Web Browser<br/>Mobile App"]
        CLI["CLI Interface"]
    end
    
    subgraph API["🌐 API LAYER (Phase 6)"]
        FastAPI["FastAPI<br>/chat, /sessions,<br/>/feedback, /health"]
        Auth["Auth & Validation<br/>Request/Response"]
    end
    
    subgraph AgentLayer["🤖 AGENT LAYER"]
        Reactive["ReactiveAgent"]
        Learning["LearningAgent"]
        LLM["LLMAgent"]
        Tools["ToolUsingAgent"]
        Orch["Orchestrator"]
    end
    
    subgraph Memory["💾 MEMORY LAYER (Phase 5)"]
        Episodic["Episodic Memory<br/>per-agent"]
        Vector["Vector Store<br/>ChromaDB/FAISS"]
    end
    
    subgraph External["🔗 EXTERNAL"]
        OpenAI["OpenAI API"]
        Anthropic["Anthropic API"]
        Ollama["Ollama Local"]
        Tools2["Web Search<br/>File I/O<br/>HTTP Calls"]
    end
    
    subgraph Obs["📊 OBSERVABILITY (Phase 7)"]
        Logs["Structured Logs<br/>structlog"]
        Metrics["Prometheus<br/>Metrics"]
        Trace["MLflow<br/>Model Tracking"]
    end
    
    subgraph Deploy["☁️ DEPLOYMENT (Phase 7)"]
        Docker["Docker<br/>Container"]
        K8S["Kubernetes<br/>Orchestration"]
        CICD["CI/CD<br/>GitHub Actions"]
    end
    
    Client --> API
    API --> Auth
    Auth --> AgentLayer
    
    AgentLayer --> Memory
    AgentLayer --> External
    
    AgentLayer --> Obs
    Memory --> Obs
    
    Docker --> K8S
    CICD --> Docker
    
    style Client fill:#e1f5ff
    style API fill:#f3e5f5
    style AgentLayer fill:#e8f5e9
    style Memory fill:#fce4ec
    style External fill:#fff3e0
    style Obs fill:#f0f4c3
    style Deploy fill:#c8e6c9
```

---

## 9. Data Flow: End-to-End Chat

![Chat Flow](chat_flow.svg)

```mermaid
sequenceDiagram
    participant User
    participant API as FastAPI<br/>Server
    participant Agent as Agent<br/>Pipeline
    participant LLM as LLM<br/>API
    participant Memory as Vector<br/>Store
    participant Metrics as Prometheus<br/>Metrics

    User->>API: POST /chat<br/>{"message": "...",<br/>"session_id": "..."}
    
    activate API
    API->>Memory: retrieve_context(message)<br/>find similar past interactions
    Memory-->>API: [context_1, context_2, ...]
    
    API->>Agent: run(message, context=...)
    activate Agent
    
    Agent->>Agent: perceive()
    Agent->>Agent: decide() → which agent?
    
    alt Uses LLMAgent
        Agent->>LLM: POST /v1/chat/completions<br/>messages=[system, history, user]
        LLM-->>Agent: response_text
    else Uses LearningAgent
        Agent->>Agent: predict() → intent
    else Uses ReactiveAgent
        Agent->>Agent: lookup() → response
    end
    
    Agent->>Agent: act() → finalize response
    Agent->>Memory: store(input, output, metadata)
    Agent-->>API: result
    deactivate Agent
    
    API->>Metrics: increment counter<br/>latency histogram
    API-->>User: POST /chat response<br/>{"response": "...",<br/>"confidence": 0.92}
    deactivate API
    
    Metrics->>Metrics: update prometheus<br/>counters/histograms
```

---

## 10. Decision Tree: Which Agent to Use?

```mermaid
graph TD
    Q1{"Is input<br/>predictable?"} 
    
    Q1 -->|Yes<br/>fixed patterns| A1["✅ ReactiveAgent<br/>rule table<br/>~1ms"]
    Q1 -->|No| Q2{"Do you have<br/>labelled training<br/>data?"}
    
    Q2 -->|Yes| A2["✅ LearningAgent<br/>classifier<br/>~5ms"]
    Q2 -->|No| Q3{"Can you<br/>afford API<br/>latency?"}
    
    Q3 -->|Yes| A3["✅ LLMAgent<br/>open-ended reasoning<br/>300-2000ms"]
    Q3 -->|No| A4["❌ Fallback to<br/>ReactiveAgent"]
    
    Q1 -->|Maybe both| A5["✅ Stack them<br/>ReactiveAgent<br/>+ LearningAgent<br/>as fallback"]
    
    style A1 fill:#c8e6c9
    style A2 fill:#c8e6c9
    style A3 fill:#c8e6c9
    style A4 fill:#ffcdd2
    style A5 fill:#ffe0b2
```

---

## 11. Technology Stack by Phase

```mermaid
graph LR
    subgraph Phase1["Phase 1"]
        P1a["Python 3.12"]
        P1b["numpy, pandas"]
        P1c["scikit-learn"]
    end
    
    subgraph Phase2["Phase 2"]
        P2a["python-dotenv"]
        P2b["openai,<br/>anthropic"]
    end
    
    subgraph Phase3["Phase 3"]
        P3a["duckduckgo-search"]
        P3b["requests"]
    end
    
    subgraph Phase4["Phase 4"]
        P4a["chromadb"]
        P4b["sentence-transformers"]
    end
    
    subgraph Phase5["Phase 5"]
        P5a["chromadb /FAISS"]
    end
    
    subgraph Phase6["Phase 6"]
        P6a["fastapi"]
        P6b["uvicorn"]
        P6c["pydantic"]
    end
    
    subgraph Phase7["Phase 7"]
        P7a["structlog"]
        P7b["prometheus-client"]
        P7c["mlflow"]
        P7d["Docker"]
        P7e["Kubernetes"]
    end
    
    Phase1 --> Phase2 --> Phase3 --> Phase4 --> Phase5 --> Phase6 --> Phase7
    
    style Phase1 fill:#c8e6c9
    style Phase2 fill:#bbdefb
    style Phase3 fill:#ffe0b2
    style Phase4 fill:#f8bbd0
    style Phase5 fill:#e1bee7
    style Phase6 fill:#b2dfdb
    style Phase7 fill:#fff9c4
```

