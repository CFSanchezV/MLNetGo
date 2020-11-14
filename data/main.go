package main

import (
	"encoding/csv"
	"fmt"
	"io"
	"math/rand"
	"net/http"
	"os"
	"strconv"
)

func main() {
	// descarga del dataset limpio y modificado a partir del dataset de la URL: "www.kaggle.com/ashkhagan/palmer-penguins-datasetalternative-iris-dataset"
	xCols := 4 // numero de columnas con datos
	// localFile, err := os.Open("penguins_raw.csv")
	resp, err := http.Get("https://drive.google.com/uc?export=download&id=1MjcjVMVFQNCXYUhTySrIWe9b8kn93QYd")

	if err != nil {
		panic("Error cargando datos de URL")
	}
	// defer localFile.Close()
	defer resp.Body.Close()

	lines := []string{}

	// r := csv.NewReader(localFile)
	r := csv.NewReader(resp.Body)
	// r.Comma = ','
	// r.FieldsPerRecord = 5

	for {
		//first line has no headers... yet
		campos, err := r.Read()
		if err == io.EOF {
			break
		}

		if err != nil {
			panic("Error al leer csv " + err.Error())
		}

		line := ""
		for i, f := range campos {
			if i < xCols {
				val := parseX(f)

				mean, stdev := colStats(i)
				val = normalize(val, mean, stdev)

				line += fmt.Sprintf("%0.5f,", val)
			} else {
				val := parseY(f)

				line += fmt.Sprintf("%s\n", val)
				lines = append(lines, line)
			}
		}

	}

	train, err := os.Create("training.csv")
	if err != nil {
		panic("Error creando archivo!")
	}
	defer train.Close()

	test, err := os.Create("testing.csv")
	if err != nil {
		panic("Error creando archivo!")
	}
	defer test.Close()

	counts := make(map[int]int, 0)
	headers := []string{"bill_length_mm,", "bill_depth_mm,", "flipper_length_mm,", "body_mass_g,", "adelie,", "gentoo,", "chinstrap"}

	for idx, i := range rand.Perm(len(lines)) {

		var c int
		if idx == 0 {
			for _, value := range headers {
				train.WriteString(value)
				test.WriteString(value)
			}
			train.WriteString("\n")
			test.WriteString("\n")
		}

		if i < 50 {
			c = 0
		} else if i > 100 {
			c = 2
		} else {
			c = 1
		}

		if counts[c] < 100 {
			train.WriteString(lines[i])
		} else {
			test.WriteString(lines[i])
		}

		counts[c]++
	}

	train.Sync()
	test.Sync()
}

// atributos del dataset, extraídos de la data utilizando Dataframes
var means = []float64{43.99, 17.16, 200.97, 4207.06}
var standardDs = []float64{5.47, 1.97, 14.02, 805.22}

func colStats(i int) (mean float64, std float64) {
	return means[i], standardDs[i]
}

func parseX(s string) float64 {
	f, err := strconv.ParseFloat(s, 64)
	if err != nil {
		panic("Error parsing float!")
	}

	return f
}

func parseY(str string) string {

	if str == "Adelie" {
		return "1.0,0.0,0.0"
	}

	if str == "Gentoo" {
		return "0.0,1.0,0.0"
	}

	if str == "Chinstrap" {
		return "0.0,0.0,1.0"
	}

	panic("Especie de pinguino desconocida usada como etiqueta")
}

func normalize(v, mean, stdev float64) float64 {
	return (v - mean) / stdev
}
