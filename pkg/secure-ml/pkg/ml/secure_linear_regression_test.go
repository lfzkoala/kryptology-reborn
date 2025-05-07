package ml

import (
	"math"
	"testing"
)

func TestSecureLinearRegression(t *testing.T) {
	// Create test data
	X := [][]float64{
		{1.0, 2.0},
		{2.0, 3.0},
		{3.0, 4.0},
		{4.0, 5.0},
	}
	y := []float64{3.0, 5.0, 7.0, 9.0}

	// Create and train model
	model := NewSecureLinearRegression()
	err := model.Train(X, y, 0.01, 1000)
	if err != nil {
		t.Fatalf("Training failed: %v", err)
	}

	// Test predictions
	testCases := []struct {
		input    []float64
		expected float64
	}{
		{[]float64{1.0, 2.0}, 3.0},
		{[]float64{2.0, 3.0}, 5.0},
		{[]float64{3.0, 4.0}, 7.0},
		{[]float64{4.0, 5.0}, 9.0},
	}

	for _, tc := range testCases {
		prediction, err := model.Predict(tc.input)
		if err != nil {
			t.Errorf("Prediction failed: %v", err)
			continue
		}

		// Allow for some numerical error
		if math.Abs(prediction-tc.expected) > 0.1 {
			t.Errorf("Expected prediction %v, got %v", tc.expected, prediction)
		}
	}
}

func TestSecureLinearRegressionInvalidInput(t *testing.T) {
	model := NewSecureLinearRegression()

	// Test with empty input
	err := model.Train([][]float64{}, []float64{}, 0.01, 1000)
	if err == nil {
		t.Error("Expected error for empty input")
	}

	// Test with mismatched dimensions
	X := [][]float64{{1.0, 2.0}}
	y := []float64{3.0, 4.0}
	err = model.Train(X, y, 0.01, 1000)
	if err == nil {
		t.Error("Expected error for mismatched dimensions")
	}

	// Test prediction with invalid input
	_, err = model.Predict([]float64{1.0})
	if err == nil {
		t.Error("Expected error for invalid prediction input")
	}
}
