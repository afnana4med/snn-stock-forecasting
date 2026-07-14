def test_installations():
    """Test GeNN and ML GeNN installations"""
    try:
        # Test GeNN
        import pygenn
        print(f"✅ GeNN (pygenn {pygenn.__version__}) imported successfully")

        # Test ML GeNN
        from ml_genn import Network, Population
        from ml_genn.neurons import LeakyIntegrateFire
        print("✅ ML GeNN imported successfully")

        # Create a simple network
        net = Network()
        with net:
            Population(LeakyIntegrateFire(), 10)
        print("✅ Created test network successfully")

        return True
    except ImportError as e:
        print(f"❌ Import Error: {e}")
        return False
    except Exception as e:
        print(f"❌ Other Error: {e}")
        return False


if __name__ == "__main__":
    print("\nTesting GeNN and ML GeNN installations...")
    success = test_installations()

    if success:
        print("\n✅ All installation tests passed!")
    else:
        print("\n❌ Some tests failed!")
