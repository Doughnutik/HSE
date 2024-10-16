package main

import (
	"context"
	"encoding/json"
	"fmt"
	"hw2/model"
	"io"
	"log"
	"net/http"
	"strings"
	"time"
)

func main() {
	var addr string = "http://localhost:%d/%s"
	res, err := http.Get(fmt.Sprintf(addr, model.Port, "version"))
	if err != nil {
		log.Fatalf("Error making /version request: %v", err)
	}
	defer res.Body.Close()

	version, _ := io.ReadAll(res.Body)
	fmt.Printf("Version: %s\n", version)

	inputString := "aGVsbG8gd29ybGQ=" // hello world
	decodeReq := model.DecodeRequest{InputString: inputString}
	reqBody, _ := json.Marshal(decodeReq)

	res, err = http.Post(fmt.Sprintf(addr, model.Port, "decode"), "application/json", strings.NewReader(string(reqBody)))
	if err != nil {
		log.Fatalf("Error making /decode request: %v", err)
	}
	defer res.Body.Close()

	var decodeRes model.DecodeResponse
	json.NewDecoder(res.Body).Decode(&decodeRes)
	fmt.Printf("Decoded string: %s\n", decodeRes.OutputString)

	ctx, cancel := context.WithTimeout(context.Background(), 15*time.Second)
	defer cancel()

	req, err := http.NewRequestWithContext(ctx, http.MethodGet, fmt.Sprintf(addr, model.Port, "hard-op"), nil)
	if err != nil {
		log.Fatalf("Error creating request: %v", err)
	}

	client := &http.Client{}
	res, err = client.Do(req)

	if err != nil {
		if strings.Contains(err.Error(), "context deadline exceeded") {
			fmt.Println("Request to /hard-op was canceled due to timeout.")
		} else {
			log.Fatalf("Error making /hard-op request: %v", err)
		}
	} else {
		defer res.Body.Close()
		body, _ := io.ReadAll(res.Body)
		fmt.Printf("Hard-op response: %s, status code: %d\n", body, res.StatusCode)
	}
}
