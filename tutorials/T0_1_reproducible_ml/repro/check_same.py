from repro.train_torch import train

if __name__ == "__main__":
    run1 = train("repro/config.yaml")
    run2 = train("repro/config.yaml")
    assert run1 == run2, f"Mismatch!\nrun1={run1}\nrun2={run2}"
    print("Success: identical init/final parameter and file checksums.")

