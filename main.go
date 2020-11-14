package main

import (
	"encoding/csv"
	"errors"
	"fmt"
	"log"
	"math"
	"math/rand"
	"os"
	"strconv"
	"sync"
	"time"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// neuralNet contains all of the information that defines a trained neural net.
type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// neuralNetConfig: the neural net architecture and learning parameters
type neuralNetConfig struct {
	inputNeurons  int
	hiddenNeurons int
	outputNeurons int
	numEpochs     int
	learningRate  float64
}

// sigmoid function as activation function
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrima: derivative of the sigmoid function for backpropagation
func sigmoidPrima(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

// Read generated/available CSV files during NN setup, concurrently
func makeTrainingData(ch chan *mat.Dense, inputs *mat.Dense, labels *mat.Dense) {
	inputs, labels = makeInputsAndLabels("data/training.csv")
	ch <- inputs
	ch <- labels
	close(ch)
}

func makeTestingData(ch chan *mat.Dense, inputs *mat.Dense, labels *mat.Dense) {
	inputs, labels = makeInputsAndLabels("data/testing.csv")
	ch <- inputs
	ch <- labels
	close(ch)
}

func main() {
	startTime := time.Now()

	//--------------------------TRAINING-------------------------------//
	// The training matrices
	ch := make(chan *mat.Dense, 2)
	var inputs *mat.Dense
	var labels *mat.Dense

	go makeTrainingData(ch, inputs, labels)
	inputs = <-ch
	labels = <-ch

	// Define the network architecture, learning rate and #epochs
	config := neuralNetConfig{
		inputNeurons:  4,
		hiddenNeurons: 3,
		outputNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	// Train the neural network.
	network := newNetwork(config)

	err := network.train(inputs, labels)
	if err != nil {
		log.Fatal(err)
	}

	//--------------------------TESTING-------------------------------//

	// Form the testing matrices
	ch2 := make(chan *mat.Dense, 2)
	var testInputs *mat.Dense
	var testLabels *mat.Dense

	go makeTestingData(ch2, testInputs, testLabels)
	testInputs = <-ch2
	testLabels = <-ch2

	// Make the predictions using the trained model.
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calculate the accuracy of the model
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ {

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)

		// Show label Row		
		fmt.Printf("Predicción: %v\n", labelRow)

		var prediction int
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Accumulate the true positive/negative count.
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}
	// Execution Time
	elapsed := time.Since(startTime)

	// Calculate the accuracy (subset accuracy)
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Mostrar predicciones para hacer comparaciones
	fmt.Println("Comparar predicciones visualmente con últimas columnas en testing.csv")
	fmt.Println("por ejemplo: 1.0,0.0,0.0 = [1 0 0]")

	// Output the Accuracy value to standard out
	fmt.Printf("\nPrecisión = %0.2f\n", accuracy)

	log.Printf("La ejecución duró %s", elapsed)
}

// NewNetwork: initialize the neural network with given config
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// train: Initialize randoms, and train neural net using backpropagate
func (nn *neuralNet) train(x, y *mat.Dense) error {

	// Initialize biases/weights with random data
	randSource := rand.NewSource(time.Now().UnixNano())
	randGen := rand.New(randSource)

	wHidden := mat.NewDense(nn.config.inputNeurons, nn.config.hiddenNeurons, nil)
	bHidden := mat.NewDense(1, nn.config.hiddenNeurons, nil)
	wOut := mat.NewDense(nn.config.hiddenNeurons, nn.config.outputNeurons, nil)
	bOut := mat.NewDense(1, nn.config.outputNeurons, nil)

	wHiddenRaw := wHidden.RawMatrix().Data
	bHiddenRaw := bHidden.RawMatrix().Data
	wOutRaw := wOut.RawMatrix().Data
	bOutRaw := bOut.RawMatrix().Data	

	// Replace matrices' RawData with random float64
	RawMatricesData := [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw}
	for _, param := range RawMatricesData {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// To store the output of the neural network.
	output := new(mat.Dense)

	// Backpropagation to adjust the weights and biases.
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Define the trained neural network.
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

// backpropagate: implement the backpropagation concurrently
func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	epochSlice := make([]int, nn.config.numEpochs)
	epochLength := len(epochSlice)

	// Wait Group for numEpochs
	var wg sync.WaitGroup
	// Channel to send errors in case of any
	ch3 := make(chan error, 2)

	wg.Add(epochLength)

	go func() {
		wg.Wait()
		close(ch3)
	}()

	var mu sync.Mutex
	// "lock/unlock: e.g. if accessing rand from goroutine:" mu.Lock() || rand.Float64() || mu.Unlock()

	for i := 0; i < epochLength; i++ {
		go func() {
			defer wg.Done()

			// Feed Forward Process
			mu.Lock()
			hiddenLayerInput := new(mat.Dense)
			hiddenLayerInput.Mul(x, wHidden)
			addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
			hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)
			mu.Unlock()

			hiddenLayerActivations := new(mat.Dense)
			applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
			hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

			outputLayerInput := new(mat.Dense)
			outputLayerInput.Mul(hiddenLayerActivations, wOut)
			addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
			outputLayerInput.Apply(addBOut, outputLayerInput)
			output.Apply(applySigmoid, outputLayerInput)

			// Backpropagation Process
			networkError := new(mat.Dense)
			networkError.Sub(y, output)

			slopeOutputLayer := new(mat.Dense)
			applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrima(v) }
			slopeOutputLayer.Apply(applySigmoidPrime, output)
			slopeHiddenLayer := new(mat.Dense)
			slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

			dOutput := new(mat.Dense)
			dOutput.MulElem(networkError, slopeOutputLayer)
			errorAtHiddenLayer := new(mat.Dense)
			errorAtHiddenLayer.Mul(dOutput, wOut.T())

			dHiddenLayer := new(mat.Dense)
			dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

			// Adjust the parameters.
			wOutAdj := new(mat.Dense)
			wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
			wOutAdj.Scale(nn.config.learningRate, wOutAdj)
			wOut.Add(wOut, wOutAdj)

			bOutAdj, err := sumAlongAxis(0, dOutput)
			if err != nil {
				ch3 <- err
			}
			bOutAdj.Scale(nn.config.learningRate, bOutAdj)
			bOut.Add(bOut, bOutAdj)

			wHiddenAdj := new(mat.Dense)
			wHiddenAdj.Mul(x.T(), dHiddenLayer)
			wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
			wHidden.Add(wHidden, wHiddenAdj)

			bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
			if err != nil {
				ch3 <- err
			}
			bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
			bHidden.Add(bHidden, bHiddenAdj)

		}()
	}

	for err := range ch3 {
		if err != nil {
			return err
		}
	}

	// Loop number of epochs using backpropagation to train the model
	// for i := 0; i < nn.config.numEpochs; i++ {

	// 	// Asynchronous Forward Propagation
	// 	hiddenLayerInput := new(mat.Dense)
	// 	hiddenLayerInput.Mul(x, wHidden)
	// 	addBHidden := func(_, col int, v float64) float64 { return v + bHidden.At(0, col) }
	// 	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	// 	hiddenLayerActivations := new(mat.Dense)
	// 	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	// 	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	// 	outputLayerInput := new(mat.Dense)
	// 	outputLayerInput.Mul(hiddenLayerActivations, wOut)
	// 	addBOut := func(_, col int, v float64) float64 { return v + bOut.At(0, col) }
	// 	outputLayerInput.Apply(addBOut, outputLayerInput)
	// 	output.Apply(applySigmoid, outputLayerInput)

	
	// 	// Asynchronous Backpropagation
	// 	networkError := new(mat.Dense)
	// 	networkError.Sub(y, output)

	// 	slopeOutputLayer := new(mat.Dense)
	// 	applySigmoidPrime := func(_, _ int, v float64) float64 { return sigmoidPrima(v) }
	// 	slopeOutputLayer.Apply(applySigmoidPrime, output)
	// 	slopeHiddenLayer := new(mat.Dense)
	// 	slopeHiddenLayer.Apply(applySigmoidPrime, hiddenLayerActivations)

	// 	dOutput := new(mat.Dense)
	// 	dOutput.MulElem(networkError, slopeOutputLayer)
	// 	errorAtHiddenLayer := new(mat.Dense)
	// 	errorAtHiddenLayer.Mul(dOutput, wOut.T())

	// 	dHiddenLayer := new(mat.Dense)
	// 	dHiddenLayer.MulElem(errorAtHiddenLayer, slopeHiddenLayer)

	// 	// Adjust the parameters.
	// 	wOutAdj := new(mat.Dense)
	// 	wOutAdj.Mul(hiddenLayerActivations.T(), dOutput)
	// 	wOutAdj.Scale(nn.config.learningRate, wOutAdj)
	// 	wOut.Add(wOut, wOutAdj)

	// 	bOutAdj, err := sumAlongAxis(0, dOutput)
	// 	if err != nil {
	// 		return err
	// 	}
	// 	bOutAdj.Scale(nn.config.learningRate, bOutAdj)
	// 	bOut.Add(bOut, bOutAdj)

	// 	wHiddenAdj := new(mat.Dense)
	// 	wHiddenAdj.Mul(x.T(), dHiddenLayer)
	// 	wHiddenAdj.Scale(nn.config.learningRate, wHiddenAdj)
	// 	wHidden.Add(wHidden, wHiddenAdj)

	// 	bHiddenAdj, err := sumAlongAxis(0, dHiddenLayer)
	// 	if err != nil {
	// 		return err
	// 	}
	// 	bHiddenAdj.Scale(nn.config.learningRate, bHiddenAdj)
	// 	bHidden.Add(bHidden, bHiddenAdj)
	// }

	return nil
}

// predict makes a prediction based on a trained neural network.
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Check to make sure that our neuralNet value is a trained model.
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("lso pesos están vacíos")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("los 'bias' están vacíos")
	}

	// Define the output of the neural network
	output := new(mat.Dense)

	// Single feed forward process, from train()
	hiddenLayerInput := new(mat.Dense)
	hiddenLayerInput.Mul(x, nn.wHidden)
	addBHidden := func(_, col int, v float64) float64 { return v + nn.bHidden.At(0, col) }
	hiddenLayerInput.Apply(addBHidden, hiddenLayerInput)

	hiddenLayerActivations := new(mat.Dense)
	applySigmoid := func(_, _ int, v float64) float64 { return sigmoid(v) }
	hiddenLayerActivations.Apply(applySigmoid, hiddenLayerInput)

	outputLayerInput := new(mat.Dense)
	outputLayerInput.Mul(hiddenLayerActivations, nn.wOut)
	addBOut := func(_, col int, v float64) float64 { return v + nn.bOut.At(0, col) }
	outputLayerInput.Apply(addBOut, outputLayerInput)
	output.Apply(applySigmoid, outputLayerInput)

	return output, nil
}

// sumAlongAxis sums a matrix along a dimension, preserving the other dimension.
func sumAlongAxis(axis int, m *mat.Dense) (*mat.Dense, error) {

	numRows, numCols := m.Dims()

	var output *mat.Dense

	switch axis {
	case 0:
		data := make([]float64, numCols)
		for i := 0; i < numCols; i++ {
			col := mat.Col(nil, i, m)
			data[i] = floats.Sum(col)
		}
		output = mat.NewDense(1, numCols, data)
	case 1:
		data := make([]float64, numRows)
		for i := 0; i < numRows; i++ {
			row := mat.Row(nil, i, m)
			data[i] = floats.Sum(row)
		}
		output = mat.NewDense(numRows, 1, data)
	default:
		return nil, errors.New("axis debe ser 0 o 1")
	}

	return output, nil
}

func makeInputsAndLabels(fileName string) (*mat.Dense, *mat.Dense) {
	// Open the given dataset file
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Read in all of the CSV records -> [][]string
	rawCSVData, err := reader.ReadAll()

	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData holds the float values usesd to form matrices.
	// fmt.Printf("Numero de inputs %d\n", len(rawCSVData))

	// 4 inputNeurons, 3 output Neurons
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// Track the current index of matrix values
	var inputsIndex int
	var labelsIndex int

	// Sequentially move the rows into a slice of floats.
	for idx, record := range rawCSVData {
		// Skip Headers' row.
		if idx == 0 {
			continue
		}

		// Loop over the float columns
		for i, val := range record {

			// Convert the value to a float.
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// Add to labelsData if relevant cols 4->(5), 5->(6) and 6->(7).
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Add the float value to the slice of floats.
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	// Show matrices with formatter

	// fmt.Println("Inputs:")
	// formatter := mat.Formatted(inputs, mat.Prefix(""))
	// fmt.Printf("%v\n", formatter)

	// fmt.Println("Labels:")
	// formatter = mat.Formatted(labels, mat.Prefix(""))
	// fmt.Printf("%v\n", formatter)

	return inputs, labels
}
