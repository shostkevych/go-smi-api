package main

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"

	"github.com/gorilla/websocket"
)

var upgrader = websocket.Upgrader{
	CheckOrigin: func(r *http.Request) bool { return true },
}

func main() {
	gpuMon := NewGPUMonitor()
	gpuMon.Start()
	defer gpuMon.Stop()

	ollamaMon := NewOllamaMonitor()
	ollamaMon.Start()
	defer ollamaMon.Stop()

	http.HandleFunc("/api/gpus", func(w http.ResponseWriter, r *http.Request) {
		metrics := gpuMon.Latest()
		if metrics == nil {
			http.Error(w, "no data yet", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(metrics)
	})

	http.HandleFunc("/api/ollama/stats", func(w http.ResponseWriter, r *http.Request) {
		stats := ollamaMon.Latest()
		if stats == nil {
			http.Error(w, "no data yet", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		json.NewEncoder(w).Encode(stats)
	})

	http.HandleFunc("/ws", func(w http.ResponseWriter, r *http.Request) {
		conn, err := upgrader.Upgrade(w, r, nil)
		if err != nil {
			log.Println("ws upgrade:", err)
			return
		}
		defer conn.Close()

		ticker := time.NewTicker(1 * time.Second)
		defer ticker.Stop()

		for range ticker.C {
			payload := struct {
				GPU    *GPUMetrics  `json:"gpu"`
				Ollama *OllamaStats `json:"ollama"`
			}{
				GPU:    gpuMon.Latest(),
				Ollama: ollamaMon.Latest(),
			}
			if err := conn.WriteJSON(payload); err != nil {
				break
			}
		}
	})

	fmt.Println("listening on :8080")
	log.Fatal(http.ListenAndServe(":8080", nil))
}
