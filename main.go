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

// neuralNet contiene la información que define una red neuronal entrenada
type neuralNet struct {
	config  neuralNetConfig
	wHidden *mat.Dense
	bHidden *mat.Dense
	wOut    *mat.Dense
	bOut    *mat.Dense
}

// neuralNetConfig: la arquitectura de la red neuronal y los parámetros de aprendizaje
type neuralNetConfig struct {
	inputNeurons  int
	hiddenNeurons int
	outputNeurons int
	numEpochs     int
	learningRate  float64
}

// sigmoid function: como función de activación
func sigmoid(x float64) float64 {
	return 1.0 / (1.0 + math.Exp(-x))
}

// sigmoidPrima: derivada de la función sigmoidea para retropropagación
func sigmoidPrima(x float64) float64 {
	return sigmoid(x) * (1.0 - sigmoid(x))
}

// Leer archivos CSV generados/disponibles durante la configuración de red,
// en simultáneo a través de canales
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
	// Las matrices de entrenamiento
	ch := make(chan *mat.Dense, 2)
	var inputs *mat.Dense
	var labels *mat.Dense

	go makeTrainingData(ch, inputs, labels)
	inputs = <-ch
	labels = <-ch

	// Definir la arquitectura de red, el factor de aprendizaje y el número de epochs
	config := neuralNetConfig{
		inputNeurons:  4,
		hiddenNeurons: 3,
		outputNeurons: 3,
		numEpochs:     5000,
		learningRate:  0.3,
	}

	// Entrenar la red neuronal
	network := newNetwork(config)

	err := network.train(inputs, labels)
	if err != nil {
		log.Fatal(err)
	}

	//--------------------------TESTING-------------------------------//

	// Formar las matrices de pruebas
	ch2 := make(chan *mat.Dense, 2)
	var testInputs *mat.Dense
	var testLabels *mat.Dense

	go makeTestingData(ch2, testInputs, testLabels)
	testInputs = <-ch2
	testLabels = <-ch2

	// Hacer predicciones usando el modelo entrenado
	predictions, err := network.predict(testInputs)
	if err != nil {
		log.Fatal(err)
	}

	// Calcular la precisión (accuracy) del modelo:
	var truePosNeg int
	numPreds, _ := predictions.Dims()
	for i := 0; i < numPreds; i++ { // i: Filas

		// Get the label.
		labelRow := mat.Row(nil, i, testLabels)

		// Mostrar fila de etiquetas (clasificaciones)
		fmt.Printf("Predicción: %v\n", labelRow)

		var prediction int // j: Columnas
		for idx, label := range labelRow {
			if label == 1.0 {
				prediction = idx
				break
			}
		}

		// Acumular el recuento verdadero positivo/negativo
		if predictions.At(i, prediction) == floats.Max(mat.Row(nil, i, predictions)) {
			truePosNeg++
		}
	}
	// Tiempo de ejecución
	elapsed := time.Since(startTime)

	// Calcular precisión: (predicciones correctas / total de predicciones realizadas)
	accuracy := float64(truePosNeg) / float64(numPreds)

	// Mostrar predicciones para hacer comparaciones
	fmt.Println("Comparar predicciones visualmente con últimas columnas en testing.csv")
	fmt.Println("por ejemplo: 1.0,0.0,0.0 = [1 0 0]")

	// Mostrar el valor de precisión
	fmt.Printf("\nPrecisión = %0.2f\n", accuracy)

	log.Printf("La ejecución duró %s", elapsed)
}

// newNetwork: inicializar red neuronal con la configuración dada
func newNetwork(config neuralNetConfig) *neuralNet {
	return &neuralNet{config: config}
}

// train: Inicializar valores aleatorios y entrenar la red neuronal mediante retropropagación
func (nn *neuralNet) train(x, y *mat.Dense) error {

	// Inicializar bias/pesos con datos aleatorios
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

	// Reemplazar los datos base de las matrices con Float64 aleatorios
	RawMatricesData := [][]float64{wHiddenRaw, bHiddenRaw, wOutRaw, bOutRaw}
	for _, param := range RawMatricesData {
		for i := range param {
			param[i] = randGen.Float64()
		}
	}

	// Almacenar la salida de la red neuronal
	output := new(mat.Dense)

	// Retropropagación para ajustar los pesos y bias
	if err := nn.backpropagate(x, y, wHidden, bHidden, wOut, bOut, output); err != nil {
		return err
	}

	// Definir la red neuronal entrenada:
	nn.wHidden = wHidden
	nn.bHidden = bHidden
	nn.wOut = wOut
	nn.bOut = bOut

	return nil
}

// Retro-propagación: implementar la propagación hacia atrás concurrentemente
func (nn *neuralNet) backpropagate(x, y, wHidden, bHidden, wOut, bOut, output *mat.Dense) error {

	epochLength := nn.config.numEpochs

	// Grupo de espera para numero de Epochs
	var wg sync.WaitGroup
	// Canal para enviar errores en caso de que surja alguno
	ch3 := make(chan error, 2)

	// Cantidad de operaciones a esperar
	wg.Add(epochLength)

	go func() {
		wg.Wait()
		close(ch3)
	}()

	var mu sync.Mutex
	// "lock/unlock: por ejemplo: si accede a rand desde una rutina go:" mu.Lock() || rand.Float64() || mu.Unlock()

	for i := 0; i < epochLength; i++ {
		go func() {
			defer wg.Done()

			// Propagación hacia adelante
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

			// proceso de Retro-propagación
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

			// Ajustar los parametros
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

	return nil
}

// predict: hacer una predicción basada en una red neuronal entrenada
func (nn *neuralNet) predict(x *mat.Dense) (*mat.Dense, error) {

	// Verificar que el valor de neuralNet sea un modelo entrenado
	if nn.wHidden == nil || nn.wOut == nil {
		return nil, errors.New("lso pesos están vacíos")
	}
	if nn.bHidden == nil || nn.bOut == nil {
		return nil, errors.New("los 'bias' están vacíos")
	}

	// Almacenar la salida de la red neuronal
	output := new(mat.Dense)

	// Proceso de propagación hacia adelante: igual a train()
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

// sumAlongAxis: sumar una matriz a lo largo de una dimensión, conservando la otra
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
	f, err := os.Open(fileName)
	if err != nil {
		log.Fatal(err)
	}
	defer f.Close()

	reader := csv.NewReader(f)
	reader.FieldsPerRecord = 7

	// Leer en todos los registros CSV -> [][]string
	rawCSVData, err := reader.ReadAll()

	if err != nil {
		log.Fatal(err)
	}

	// inputsData and labelsData contienen los valores flotantes que se utilizan para formar matrices
	// fmt.Printf("Numero de inputs en dataset: %d\n", len(rawCSVData))

	// 4 neuronas de entrada, 3 neuronas de salida (etiquetas o clasificación)
	inputsData := make([]float64, 4*len(rawCSVData))
	labelsData := make([]float64, 3*len(rawCSVData))

	// Seguimiento del índice actual de valores de matriz
	var inputsIndex int
	var labelsIndex int

	// Mover secuencialmente las filas en un slice de flotantes
	for idx, record := range rawCSVData {
		// Omitir fila de encabezados
		if idx == 0 {
			continue
		}

		// Bucle sobre las columnas (flotantes)
		for i, val := range record {

			// Convertir el valor en un flotante
			parsedVal, err := strconv.ParseFloat(val, 64)
			if err != nil {
				log.Fatal(err)
			}

			// columnas = {0,1,2,3,4,5,6}
			// Agregar a Data de Etiquetas si corresponde: columnas 4, 5, y 6.
			if i == 4 || i == 5 || i == 6 {
				labelsData[labelsIndex] = parsedVal
				labelsIndex++
				continue
			}

			// Agregar el valor flotante al segmento de flotantes
			inputsData[inputsIndex] = parsedVal
			inputsIndex++
		}
	}

	inputs := mat.NewDense(len(rawCSVData), 4, inputsData)
	labels := mat.NewDense(len(rawCSVData), 3, labelsData)

	// Ver matrices a utilizar con mat.Formatted

	// fmt.Println("Inputs:")
	// formatter := mat.Formatted(inputs, mat.Prefix(""))
	// fmt.Printf("%v\n", formatter)

	// fmt.Println("Labels:")
	// formatter = mat.Formatted(labels, mat.Prefix(""))
	// fmt.Printf("%v\n", formatter)

	return inputs, labels
}
