def test_genn_installation():
    try:
        from pygenn import GeNNModel
        print("PyGeNN imported successfully")

        # Try to create a simple GeNN model
        GeNNModel("float", "test_model")
        print("GeNN model created successfully")
        return True
    except Exception as e:
        print(f"Error testing GeNN installation: {e}")
        return False


def test_ml_genn():
    try:
        from ml_genn import Network, Population
        from ml_genn.neurons import LeakyIntegrateFire
        print("ML GeNN imported successfully")

        # Create a simple network
        net = Network()
        with net:
            Population(LeakyIntegrateFire(), 10)
        print("ML GeNN network created successfully")
        return True
    except Exception as e:
        print(f"Error testing ML GeNN: {e}")
        return False


if __name__ == "__main__":
    print("Testing GeNN installation...")
    genn_ok = test_genn_installation()

    print("\nTesting ML GeNN installation...")
    ml_genn_ok = test_ml_genn()

    if genn_ok and ml_genn_ok:
        print("\n✅ All tests passed!")
    else:
        print("\n❌ Some tests failed!")
