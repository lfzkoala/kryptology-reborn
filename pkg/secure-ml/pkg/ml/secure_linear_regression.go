package ml

import (
	"errors"
)

// SecureLinearRegression represents a linear regression model that can be trained and used for predictions
// while keeping the data encrypted
type SecureLinearRegression struct {
	weights []float64
	bias    float64
}

// NewSecureLinearRegression creates a new secure linear regression model
func NewSecureLinearRegression() *SecureLinearRegression {
	return &SecureLinearRegression{
		weights: make([]float64, 0),
		bias:    0.0,
	}
}

// Train performs secure training of the linear regression model
// In a real implementation, this would use homomorphic encryption
func (slr *SecureLinearRegression) Train(X [][]float64, y []float64, learningRate float64, epochs int) error {
	if len(X) == 0 || len(X) != len(y) {
		return errors.New("invalid input data")
	}

	// Initialize weights if not already done
	if len(slr.weights) == 0 {
		slr.weights = make([]float64, len(X[0]))
	}

	// Training loop
	for epoch := 0; epoch < epochs; epoch++ {
		for i := range X {
			// Compute prediction
			prediction := slr.predict(X[i])

			// Compute error
			error := y[i] - prediction

			// Update weights and bias
			for j := range slr.weights {
				slr.weights[j] += learningRate * error * X[i][j]
			}
			slr.bias += learningRate * error
		}
	}

	return nil
}

// Predict makes a prediction using the trained model
// In a real implementation, this would use homomorphic encryption
func (slr *SecureLinearRegression) Predict(x []float64) (float64, error) {
	if len(x) != len(slr.weights) {
		return 0, errors.New("input dimension mismatch")
	}
	return slr.predict(x), nil
}

// predict is an internal method that computes the prediction
func (slr *SecureLinearRegression) predict(x []float64) float64 {
	prediction := slr.bias
	for i := range x {
		prediction += slr.weights[i] * x[i]
	}
	return prediction
}

// GetWeights returns the current model weights
func (slr *SecureLinearRegression) GetWeights() []float64 {
	return slr.weights
}

// GetBias returns the current model bias
func (slr *SecureLinearRegression) GetBias() float64 {
	return slr.bias
}
