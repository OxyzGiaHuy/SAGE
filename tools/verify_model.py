import torch
import sys
import os
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Add the project root to sys.path to ensure 'sage' package is findable
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(project_root)

def run_verification():
    print("=" * 60)
    print("SAGE Repository Structure Verification")
    print("=" * 60)
    
    try:
        # 1. Test Imports
        print("\n[1/4] Testing Imports...")
        from sage.networks.convnext_transformer_unet import create_convnext_transformer_unet
        from sage.components.sage_layer import SageLayer
        print("✓ Successfully imported SAGE components and model factory.")

        # 2. Test Model Instantiation (SAGE Mode)
        print("\n[2/4] Instantiating SAGE Model...")
        sage_config = {
            'top_k': 2,
            'load_balance_factor': 0.01,
            'alpha': 0.9,
            'gating_type': 'softmax' 
        }
        
        model = create_convnext_transformer_unet(
            num_classes=3,
            img_size=224,
            convnext_variant='tiny', # Use tiny for faster verification
            vit_variant='base',
            num_transformer_layers=2,
            sage_config=sage_config 
        )
        print(f"✓ SAGE instantiated successfully.")
        
        # 3. Model Information
        info = model.get_model_info()
        print(f"  - Total Parameters: {info['total_parameters']:,}")
        print(f"  - Input Size: {info['img_size']}x{info['img_size']}")
        
        # 4. Forward Pass Test
        print("\n[3/4] Testing Forward Pass...")
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        dummy_input = torch.randn(1, 3, 224, 224).to(device)
        print(f"  - Input shape: {dummy_input.shape}")
        
        with torch.no_grad():
            output = model(dummy_input)
            
        print(f"✓ Forward pass successful.")
        print(f"  - Output shape: {output.shape}")
        
        # 5. Check expert injection
        print("\n[4/4] Verifying SAGE Layer Injection...")
        sage_layers = [m for m in model.modules() if isinstance(m, SageLayer)]
        print(f"✓ Found {len(sage_layers)} SageLayers injected into the architecture.")
        
        print("\n" + "=" * 60)
        print("✨ VERIFICATION SUCCESSFUL: Repository is functional!")
        print("=" * 60)
        return True

    except Exception as e:
        print("\n" + "!" * 60)
        print(f"❌ VERIFICATION FAILED: {str(e)}")
        print("!" * 60)
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = run_verification()
    sys.exit(0 if success else 1)
