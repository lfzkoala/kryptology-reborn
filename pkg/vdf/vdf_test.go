package vdf

import (
	"bytes"
	"crypto/sha256"
	"github.com/coinbase/kryptology/pkg/core/iqc"
	"golang.org/x/crypto/sha3"
	"math/big"
	"testing"
	"time"
)

func TestVDF(t *testing.T) {
	t.Run("New VDF", testNewVDF)
	t.Run("Execute VDF", testExecuteVDF)
	t.Run("Verify VDF", testVerifyVDF)
	t.Run("Execute Iteration", testExecuteIteration)
	t.Run("Verify Iteration", testVerifyIteration)
	t.Run("IsFinished and GetOutput", testIsFinishedAndGetOutput)
}

func testNewVDF(t *testing.T) {
	difficulty := uint32(1000)
	input := sha3.Sum256([]byte("test input"))
	vdf := New(difficulty, input)

	if vdf.difficulty != difficulty {
		t.Errorf("Expected difficulty %d, got %d", difficulty, vdf.difficulty)
	}

	if vdf.input != input {
		t.Errorf("Expected input %v, got %v", input, vdf.input)
	}

	if vdf.outputChan == nil {
		t.Error("Output channel should not be nil")
	}
}

func testExecuteVDF(t *testing.T) {
	difficulty := uint32(100)
	input := sha3.Sum256([]byte("test input"))
	vdf := New(difficulty, input)

	go vdf.Execute()

	select {
	case output := <-vdf.GetOutputChannel():
		if len(output) != 516 {
			t.Errorf("Expected output length 516, got %d", len(output))
		}
	case <-time.After(5 * time.Second):
		t.Error("VDF execution timed out")
	}
}

func testVerifyVDF(t *testing.T) {
	difficulty := uint32(100)
	input := sha3.Sum256([]byte("test input"))
	vdf := New(difficulty, input)

	vdf.Execute()
	output := vdf.GetOutput()

	if !vdf.Verify(output) {
		t.Error("VDF verification failed")
	}
}

func testExecuteIteration(t *testing.T) {
	difficulty := uint32(100) // Use a small difficulty for faster testing
	input := sha256.Sum256([]byte("test input"))
	vdf := New(difficulty, input)

	x_blob := make([]byte, 258)
	copy(x_blob, []byte("iteration input"))

	go vdf.ExecuteIteration(x_blob)

	select {
	case output := <-vdf.GetOutputChannel():
		if len(output) != 516 {
			t.Errorf("Expected output length 516, got %d", len(output))
		}
	case <-time.After(5 * time.Second):
		t.Error("VDF iteration execution timed out")
	}
}

func testVerifyIteration(t *testing.T) {
	difficulty := uint32(100) // Use a small difficulty for faster testing
	input := sha256.Sum256([]byte("test input"))
	vdf := New(difficulty, input)

	// Generate initial x_blob
	var x_blob [258]byte
	initialClassGroup := iqc.NewClassGroupFromAbDiscriminant(big.NewInt(2), big.NewInt(1), iqc.CreateDiscriminant(input[:], 2048))
	if initialClassGroup == nil {
		t.Fatal("Failed to create initial class group")
	}
	copy(x_blob[:], initialClassGroup.Serialize())

	// Execute iteration
	vdf.ExecuteIteration(x_blob[:])
	output := vdf.GetOutput()

	// Verify iteration
	if !vdf.VerifyIteration(x_blob, output) {
		t.Error("VDF iteration verification failed")
	}
}

func testIsFinishedAndGetOutput(t *testing.T) {
	difficulty := uint32(100) // Use a small difficulty for faster testing
	input := sha256.Sum256([]byte("test input"))
	vdf := New(difficulty, input)

	if vdf.IsFinished() {
		t.Error("VDF should not be finished before execution")
	}

	vdf.Execute()

	if !vdf.IsFinished() {
		t.Error("VDF should be finished after execution")
	}

	output := vdf.GetOutput()
	if bytes.Equal(output[:], make([]byte, 516)) {
		t.Error("Output should not be all zeros after execution")
	}
}
