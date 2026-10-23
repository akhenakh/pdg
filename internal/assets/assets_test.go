package assets

import "testing"

// The model and lexicons must be compiled in: pdg has no runtime data files.
func TestEmbedded(t *testing.T) {
	if len(Model) < 1_000_000 {
		t.Fatalf("model is not embedded (%d bytes)", len(Model))
	}
	if len(Gold) == 0 || len(Silver) == 0 {
		t.Fatal("lexicons are not embedded")
	}
	if len(Config) == 0 {
		t.Fatal("config is not embedded")
	}
}
