graph TD
    %% Input Layer
    A[Game Environment Input] --> B[OptimizedStreetFighterVisionWrapper.step]
    B --> C[Frame Processing & Feature Extraction]
    
    %% Frame Processing
    C --> D[OptimizedFrameProcessor.process]
    D --> D1[RGB to Grayscale Conversion]
    D1 --> D2[Resize to 84x84 using INTER_AREA]
    D2 --> D3[Memory-Optimized Output]
    
    %% Feature Extraction
    C --> E[EBTEnhancedFeatureTracker.get_features]
    E --> E1[Health History Processing]
    E1 --> E2[Action & Combo Tracking]
    E2 --> E3[Ensure Feature Dimension: 24D Vector]
    
    %% Observation Building
    D3 --> F[_build_observation]
    E3 --> F
    F --> F1[Visual: 1x84x84 Grayscale]
    F --> F2[Vector: 3x24 Feature History]
    
    %% CNN Feature Extraction
    F1 --> G[OptimizedStreetFighterCNN.forward]
    F2 --> G
    
    %% CNN Visual Path
    G --> G1[Visual CNN Branch]
    G1 --> G1a[Conv2d: 1→16, k=8, s=4]
    G1a --> G1b[ReLU + BatchNorm2d]
    G1b --> G1c[Conv2d: 16→32, k=4, s=2]
    G1c --> G1d[ReLU + BatchNorm2d]
    G1d --> G1e[Conv2d: 32→64, k=3, s=2]
    G1e --> G1f[ReLU + BatchNorm2d]
    G1f --> G1g[AdaptiveAvgPool2d to 2x2]
    G1g --> G1h[Flatten → Visual Features]
    
    %% CNN Vector Path
    G --> G2[Vector Processing Branch]
    G2 --> G2a[Linear Embedding: 24→32]
    G2a --> G2b[LayerNorm + Dropout]
    G2b --> G2c[GRU: seq_len=3, hidden=32]
    G2c --> G2d[Take Last Output]
    G2d --> G2e[Linear: 32→16]
    
    %% CNN Fusion
    G1h --> G3[Feature Fusion]
    G2e --> G3
    G3 --> G3a[Concatenate Visual + Vector]
    G3a --> G3b[Linear: combined→256]
    G3b --> G3c[ReLU + LayerNorm + Dropout]
    G3c --> G3d[Linear: 256→128]
    G3d --> G3e[ReLU → Final Features]
    
    %% EBT Preparation
    G3e --> G4[EBT Projection]
    G4 --> G4a[Linear: 128→128 for EBT]
    
    %% Sequence Tracking
    E --> H[EBTSequenceTracker.add_step]
    H --> H1[Update State Sequence]
    H1 --> H2[Update Action Sequence]
    H2 --> H3[Update Reward Sequence]
    H3 --> H4[Update Feature Sequence]
    H4 --> H5[Create Combined Features]
    
    %% EBT Sequence Processing
    H --> I[EBTSequenceTracker.get_sequence_tensor]
    I --> I1[Pad to Sequence Length: 8]
    I1 --> I2[Stack to Tensor: 1x8x24]
    
    %% Energy-Based Transformer
    I2 --> J[EnergyBasedTransformer.forward]
    J --> J1[Input Projection: 24→128]
    J1 --> J2[EBTPositionalEncoding]
    J2 --> J2a[Sin/Cos Positional Encoding]
    J2a --> J2b[Add Position to Input]
    
    %% EBT Transformer Blocks
    J2b --> J3[EBTTransformerBlock Loop: 3 layers]
    J3 --> J3a[EBTMultiHeadAttention]
    J3a --> J3a1[Query/Key/Value Projections]
    J3a1 --> J3a2[4-Head Attention: d_k=32]
    J3a2 --> J3a3[Scaled Dot-Product Attention]
    J3a3 --> J3a4[Causal Mask Application]
    J3a4 --> J3a5[Attention Weights + Dropout]
    J3a5 --> J3a6[Context Calculation]
    J3a6 --> J3a7[Output Projection + Residual]
    
    J3a7 --> J3b[Feed Forward Network]
    J3b --> J3b1[Linear: 128→256]
    J3b1 --> J3b2[GELU Activation]
    J3b2 --> J3b3[Dropout]
    J3b3 --> J3b4[Linear: 256→128]
    J3b4 --> J3b5[Dropout + Residual + LayerNorm]
    
    %% EBT Energy Calculation
    J3b5 --> J4[Energy Head Processing]
    J4 --> J4a[Linear: 128→64]
    J4a --> J4b[GELU + Dropout]
    J4b --> J4c[Linear: 64→1]
    J4c --> J4d[Per-token Energies]
    
    %% EBT Context Aggregation
    J3b5 --> J5[Context Aggregation]
    J5 --> J5a[Linear: 128→64]
    J5a --> J5b[GELU]
    J5b --> J5c[Linear: 64→128]
    J5c --> J5d[Attention Score Calculation]
    J5d --> J5e[Softmax Attention Weights]
    J5e --> J5f[Weighted Energy Aggregation]
    J5f --> J5g[Sequence-level Energy]
    
    %% Verifier Processing
    G3e --> K[OptimizedStreetFighterVerifier.forward]
    G4a --> K
    I2 --> K
    
    K --> K1[Context Features Processing]
    K1 --> K2[Action Embedding]
    K2 --> K2a[Linear: action_dim→32]
    K2a --> K2b[ReLU + LayerNorm]
    
    %% Verifier EBT Integration
    K --> K3[EBT Integration Branch]
    K3 --> K3a[Call EnergyBasedTransformer]
    K3a --> J
    J4d --> K3b[Extract Sequence Energy]
    J3b5 --> K3c[Extract Last Representation]
    
    %% Verifier Energy Network
    K1 --> K4[Energy Network Input]
    K2b --> K4
    K3c --> K4
    K4 --> K4a[Concatenate All Features]
    K4a --> K4b[Linear: combined→128]
    K4b --> K4c[ReLU + LayerNorm]
    K4c --> K4d[Linear: 128→64]
    K4d --> K4e[ReLU]
    K4e --> K4f[Linear: 64→1]
    K4f --> K4g[Energy Scaling × 0.5]
    
    %% EBT Energy Contribution
    K3b --> K5[EBT Energy Contribution]
    K5 --> K5a[Weight: 0.2 × EBT Energy]
    K5a --> K6[Combine with Base Energy]
    K4g --> K6
    K6 --> K7[Clamp Energy: -5.0 to 5.0]
    
    %% Agent Prediction
    K7 --> L[OptimizedEnergyBasedAgent.predict]
    L --> L1[Initialize Candidate Action]
    L1 --> L1a[Softmax Normalization]
    L1a --> L1b[Require Gradients]
    
    %% Agent Thinking Loop
    L1b --> L2[Thinking Loop: 2 steps]
    L2 --> L2a[Energy Calculation via Verifier]
    L2a --> K
    K7 --> L2b[Gradient Calculation]
    L2b --> L2c[Gradient Clipping: 0.5]
    L2c --> L2d[Candidate Action Update]
    L2d --> L2e[Learning Rate: 0.03]
    L2e --> L2f[Softmax Renormalization]
    L2f --> L2g[Early Stop Check]
    L2g --> L2h{Continue Thinking?}
    L2h -->|Yes| L2a
    L2h -->|No| L3[Final Action Selection]
    
    %% Final Action Selection
    L3 --> L3a[Final Softmax]
    L3a --> L3b[Add Epsilon: 1e-8]
    L3b --> L3c[Renormalize]
    L3c --> L3d{Deterministic?}
    L3d -->|Yes| L3e[Argmax Selection]
    L3d -->|No| L3f[Multinomial Sampling]
    L3e --> L4[Return Action Index]
    L3f --> L4
    
    %% Reward Calculation
    L4 --> M[OptimizedRewardCalculator.calculate_reward]
    M --> M1[Health Differential Analysis]
    M1 --> M1a[Player Damage Taken]
    M1a --> M1b[Opponent Damage Dealt]
    M1b --> M1c[Damage Reward: ×2.0 scale]
    
    M --> M2[Combat Action Bonus]
    M2 --> M2a[Attack Actions: +0.1]
    M2a --> M2b[Idle Penalty: -0.02]
    
    M --> M3[Round Completion]
    M3 --> M3a[Win Bonus: +10.0]
    M3a --> M3b[Loss Penalty: -5.0]
    M3b --> M3c[Time Penalty: -0.05]
    
    %% Experience Buffer
    M --> N[EBTEnhancedExperienceBuffer.add_experience]
    N --> N1[Quality Score Evaluation]
    N1 --> N2{Quality ≥ Threshold?}
    N2 -->|Yes| N3[Add to Good Experiences]
    N2 -->|No| N4[Add to Bad Experiences]
    N3 --> N5[Golden Buffer Check]
    N5 --> N6[Store EBT Sequence]
    
    %% Style Classes
    classDef input fill:#e1f5fe
    classDef processing fill:#f3e5f5
    classDef cnn fill:#e8f5e8
    classDef ebt fill:#fff3e0
    classDef verifier fill:#fce4ec
    classDef agent fill:#f1f8e9
    classDef reward fill:#fff8e1
    classDef buffer fill:#e0f2f1
    
    class A,B,C input
    class D,D1,D2,D3,E,E1,E2,E3,F,F1,F2 processing
    class G,G1,G1a,G1b,G1c,G1d,G1e,G1f,G1g,G1h,G2,G2a,G2b,G2c,G2d,G2e,G3,G3a,G3b,G3c,G3d,G3e,G4,G4a cnn
    class H,H1,H2,H3,H4,H5,I,I1,I2,J,J1,J2,J2a,J2b,J3,J3a,J3a1,J3a2,J3a3,J3a4,J3a5,J3a6,J3a7,J3b,J3b1,J3b2,J3b3,J3b4,J3b5,J4,J4a,J4b,J4c,J4d,J5,J5a,J5b,J5c,J5d,J5e,J5f,J5g ebt
    class K,K1,K2,K2a,K2b,K3,K3a,K3b,K3c,K4,K4a,K4b,K4c,K4d,K4e,K4f,K4g,K5,K5a,K6,K7 verifier
    class L,L1,L1a,L1b,L2,L2a,L2b,L2c,L2d,L2e,L2f,L2g,L2h,L3,L3a,L3b,L3c,L3d,L3e,L3f,L4 agent
    class M,M1,M1a,M1b,M1c,M2,M2a,M2b,M3,M3a,M3b,M3c reward
    class N,N1,N2,N3,N4,N5,N6 buffer