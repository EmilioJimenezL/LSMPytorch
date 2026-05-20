# AI Agent Instructions for LSM Multi-Model Implementation

## CRITICAL: Before Starting Any Implementation

1. **Read these documents in order:**
   - AI_AGENT_QUICK_START_GUIDE.md
   - AI_AGENT_PROMPT_MultiModel_Implementation.md (FULL READ)
   - AI_AGENT_CODE_TEMPLATES.md (reference while coding)
   - TCN_vs_3DCNN_Detailed_Analysis.md (for technical context)

2. **Execution Order - STRICT:**
   - Phase 1: Setup & Environment (don't skip!)
   - Phase 2: Model Implementation (copy from templates)
   - Phase 3: Training Infrastructure
   - Phase 4: Comparison Framework
   - Phase 5: Testing & Documentation
   - **Test after each phase before proceeding**

3. **Using the Documents:**
   - Main spec: AI_AGENT_PROMPT_MultiModel_Implementation.md
   - Copy code from: AI_AGENT_CODE_TEMPLATES.md
   - Reference for technical questions: TCN_vs_3DCNN_Detailed_Analysis.md
   - Broader context: LSM_Architecture_Overhaul.md

4. **Project Constraints:**
   - RTX 3060 Mobile: 6GB VRAM available
   - Input shape: (batch, 85, 135)
   - Output shape: (batch, 330)
   - Models must be <15MB for iOS
   - Training time: 10-20 hours expected

5. **If Stuck:**
   - Check debugging section in AI_AGENT_PROMPT_MultiModel_Implementation.md
   - Run smoke tests: python test_models_smoke.py
   - Check VRAM: nvidia-smi
   - Review checkpoint saving logic

6. **Success Criteria by Phase:**
   - Phase 1: Smoke tests pass, 1D CNN refactored
   - Phase 2: All 3 models create and forward-pass
   - Phase 3: train_multimodel.py works with all models
   - Phase 4: compare_models.py generates report
   - Phase 5: Unit tests pass, README updated