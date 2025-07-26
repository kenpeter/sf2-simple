graph TD
    %% Environment Flow
    A[Environment Reset] --> B[StreetFighterFixedWrapper.reset]
    B --> C[HealthDetector.get_health]
    C --> D[SimplifiedFeatureTracker.update]
    D --> E[_build_observation]
    E --> F[Return Initial Observation]

    %% Episode Step Flow
    G[Agent.predict] --> H[SimpleAgent.predict]
    H --> I[SimpleCNN.forward - Visual Processing]
    I --> J[SimpleCNN.forward - Vector Processing]
    J --> K[SimpleCNN.forward - Feature Fusion]
    K --> L[SimpleVerifier.forward]
    L --> M[Energy Calculation]
    M --> N[Thinking Loop - Gradient Descent]
    N --> O[Action Selection]

    %% Environment Step Flow
    O --> P[StreetFighterFixedWrapper.step]
    P --> Q[Action Conversion]
    Q --> R[Environment.step]
    R --> S[HealthDetector.get_health - Multi-method]
    S --> T[FixedRewardCalculator.calculate_reward]
    T --> U[SimplifiedFeatureTracker.update]
    U --> V[_build_observation]
    V --> W[Return Step Results]

    %% Training Flow
    W --> X[FixedExperienceBuffer.add_experience]
    X --> Y[Training Step Trigger]
    Y --> Z[FixedTrainer.train_step]
    Z --> AA[Sample Balanced Batch]
    AA --> AB[Process Good Batch]
    AA --> AC[Process Bad Batch]
    AB --> AD[SimpleCNN.forward - Good Examples]
    AC --> AE[SimpleCNN.forward - Bad Examples]
    AD --> AF[SimpleVerifier.forward - Good Energy]
    AE --> AG[SimpleVerifier.forward - Bad Energy]
    AF --> AH[Contrastive Loss Calculation]
    AG --> AH
    AH --> AI[Backward Pass]
    AI --> AJ[Parameter Update]

    %% Health Detection Detail
    S --> S1[extract_health_from_memory]
    S --> S2[extract_health_from_ram]
    S --> S3[extract_health_from_visual]
    S1 --> S4[_validate_health_reading]
    S2 --> S4
    S3 --> S4
    S4 --> S5[Health History Update]

    %% CNN Architecture Detail
    I --> I1[Visual CNN - Conv2d Layers]
    I1 --> I2[AdaptiveAvgPool2d]
    I2 --> I3[Flatten]
    J --> J1[Vector Processor - Linear Layers]
    J1 --> J2[ReLU Activations]
    I3 --> K1[Feature Concatenation]
    J2 --> K1
    K1 --> K2[Fusion Network - Linear + Dropout]

    %% Verifier Detail
    L --> L1[Action Embedding]
    L --> L2[Context Features from CNN]
    L1 --> L3[Feature Concatenation]
    L2 --> L3
    L3 --> L4[Energy Network - Multiple Linear Layers]
    L4 --> L5[Energy Scaling]

    %% Color coding for different components
    classDef envClass fill:#e1f5fe
    classDef modelClass fill:#f3e5f5
    classDef trainClass fill:#e8f5e8
    classDef healthClass fill:#fff3e0
    
    class A,B,P,Q,R,W envClass
    class H,I,J,K,L,M,N,O,I1,I2,I3,J1,J2,K1,K2,L1,L2,L3,L4,L5 modelClass
    class X,Y,Z,AA,AB,AC,AD,AE,AF,AG,AH,AI,AJ trainClass
    class C,S,S1,S2,S3,S4,S5 healthClass