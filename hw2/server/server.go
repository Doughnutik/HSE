package main

import (
	"context"
	"encoding/base64"
	"encoding/json"
	"fmt"
	"hw2/model"
	"log"
	"math/rand"
	"net/http"
	"os"
	"os/signal"
	"syscall"
	"time"
)

func main() {
	mux := http.NewServeMux()

	mux.HandleFunc("/version", func(w http.ResponseWriter, r *http.Request) {
		version := "v1.0.0"
		fmt.Fprint(w, version)
	})

	mux.HandleFunc("/decode", func(w http.ResponseWriter, r *http.Request) {
		var req model.DecodeRequest
		err := json.NewDecoder(r.Body).Decode(&req)
		if err != nil {
			http.Error(w, "Invalid request", http.StatusBadRequest)
			return
		}

		data, err := base64.StdEncoding.DecodeString(req.InputString)
		if err != nil {
			fmt.Printf("Error while decoding base64: %s\n", err)
			http.Error(w, "Invalid base64 string", http.StatusBadRequest)
			return
		}

		res := model.DecodeResponse{OutputString: string(data)}
		err = json.NewEncoder(w).Encode(res)
		if err != nil {
			fmt.Printf("Error while writing response: %s\n", err)
			http.Error(w, "Error while writing response", http.StatusInternalServerError)
			return
		}
	})

	mux.HandleFunc("/hard-op", func(w http.ResponseWriter, r *http.Request) {
		dur := time.Duration(10 + rand.Intn(11))
		time.Sleep(dur * time.Second)

		if rand.Intn(2) == 0 {
			w.WriteHeader(http.StatusInternalServerError)
			fmt.Fprintln(w, "500 Internal Server Error")
		} else {
			fmt.Fprintln(w, "200 OK")
		}
	})

	server := &http.Server{
		Addr:    fmt.Sprintf(":%d", model.Port),
		Handler: mux,
	}

	quit := make(chan os.Signal, 1)
	signal.Notify(quit, syscall.SIGINT, syscall.SIGTERM)

	go func() {
		if err := server.ListenAndServe(); err != nil && err != http.ErrServerClosed {
			log.Fatalf("ListenAndServe(): %v", err)
		}
	}()
	log.Printf("Server started on %d port", model.Port)

	<-quit
	log.Println("Server is shutting down...")

	shutdownCtx, shutdownRelease := context.WithTimeout(context.Background(), 10*time.Second)
	defer shutdownRelease()

	if err := server.Shutdown(shutdownCtx); err != nil {
		log.Fatalf("Server forced to shutdown: %v", err)
	}
	log.Println("Server exited")
}
