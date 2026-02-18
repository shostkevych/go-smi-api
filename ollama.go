package main

import (
	"encoding/json"
	"fmt"
	"net/http"
	"os"
	"strings"
	"sync"
	"time"
)

// Ollama API response types

type ollamaPsResponse struct {
	Models []ollamaPsModel `json:"models"`
}

type ollamaPsModel struct {
	Name      string             `json:"name"`
	Model     string             `json:"model"`
	Size      int64              `json:"size"`
	Digest    string             `json:"digest"`
	Details   ollamaModelDetails `json:"details"`
	ExpiresAt string             `json:"expires_at"`
	SizeVRAM  int64              `json:"size_vram"`
}

type ollamaModelDetails struct {
	Family            string `json:"family"`
	ParameterSize     string `json:"parameter_size"`
	QuantizationLevel string `json:"quantization_level"`
}

type ollamaTagsResponse struct {
	Models []ollamaTagModel `json:"models"`
}

type ollamaTagModel struct {
	Name    string             `json:"name"`
	Size    int64              `json:"size"`
	Details ollamaModelDetails `json:"details"`
}

type ollamaShowResponse struct {
	ModelInfo  map[string]interface{} `json:"model_info"`
	Details    ollamaModelDetails     `json:"details"`
	Parameters string                `json:"parameters"`
}

type ollamaVersionResponse struct {
	Version string `json:"version"`
}

// Public data model

type KVCacheInfo struct {
	DType         string  `json:"dtype"`
	BytesPerToken int     `json:"bytes_per_token"`
	MaxSizeBytes  int64   `json:"max_size_bytes"`
	MaxSizeMiB    float64 `json:"max_size_mib"`
}

type VRAMBreakdown struct {
	TotalBytes      int64 `json:"total_bytes"`
	WeightsEstBytes int64 `json:"weights_est_bytes"`
	KVCacheMaxBytes int64 `json:"kv_cache_max_bytes"`
}

type RunningModel struct {
	Name          string        `json:"name"`
	SizeVRAMBytes int64         `json:"size_vram_bytes"`
	ParameterSize string        `json:"parameter_size"`
	Quantization  string        `json:"quantization"`
	Family        string        `json:"family"`
	ExpiresAt     string        `json:"expires_at"`
	ContextWindow int           `json:"context_window"`
	KVCache       KVCacheInfo   `json:"kv_cache"`
	VRAM          VRAMBreakdown `json:"vram"`
}

type OllamaStats struct {
	Timestamp            string         `json:"timestamp"`
	Running              bool           `json:"running"`
	Version              string         `json:"version"`
	RunningModels        []RunningModel `json:"running_models"`
	AvailableModelsCount int            `json:"available_models_count"`
	TotalDiskUsageBytes  int64          `json:"total_disk_usage_bytes"`
}

// Monitor

type OllamaMonitor struct {
	mu        sync.RWMutex
	latest    *OllamaStats
	stopCh    chan struct{}
	host      string
	client    *http.Client
	showCache map[string]*ollamaShowResponse
}

func NewOllamaMonitor() *OllamaMonitor {
	host := os.Getenv("OLLAMA_HOST")
	if host == "" {
		host = "http://localhost:11434"
	}
	if !strings.HasPrefix(host, "http") {
		host = "http://" + host
	}
	return &OllamaMonitor{
		stopCh:    make(chan struct{}),
		host:      host,
		client:    &http.Client{Timeout: 5 * time.Second},
		showCache: make(map[string]*ollamaShowResponse),
	}
}

func (m *OllamaMonitor) Start() {
	m.poll()
	go func() {
		ticker := time.NewTicker(5 * time.Second)
		defer ticker.Stop()
		for {
			select {
			case <-ticker.C:
				m.poll()
			case <-m.stopCh:
				return
			}
		}
	}()
}

func (m *OllamaMonitor) Stop() {
	close(m.stopCh)
}

func (m *OllamaMonitor) Latest() *OllamaStats {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.latest
}

func (m *OllamaMonitor) poll() {
	stats := m.fetch()
	m.mu.Lock()
	m.latest = stats
	m.mu.Unlock()
}

func (m *OllamaMonitor) fetch() *OllamaStats {
	stats := &OllamaStats{
		Timestamp:     time.Now().UTC().Format(time.RFC3339),
		RunningModels: []RunningModel{},
	}

	// Liveness
	resp, err := m.client.Get(m.host + "/")
	if err != nil {
		return stats
	}
	resp.Body.Close()
	stats.Running = true

	// Version
	var ver ollamaVersionResponse
	if err := m.getJSON("/api/version", &ver); err == nil {
		stats.Version = ver.Version
	}

	// Available models
	var tags ollamaTagsResponse
	if err := m.getJSON("/api/tags", &tags); err == nil {
		stats.AvailableModelsCount = len(tags.Models)
		for _, t := range tags.Models {
			stats.TotalDiskUsageBytes += t.Size
		}
	}

	// Running models
	var ps ollamaPsResponse
	if err := m.getJSON("/api/ps", &ps); err != nil {
		return stats
	}

	kvDtype := os.Getenv("OLLAMA_KV_CACHE_TYPE")
	if kvDtype == "" {
		kvDtype = "f16"
	}

	for _, model := range ps.Models {
		rm := RunningModel{
			Name:          model.Name,
			SizeVRAMBytes: model.SizeVRAM,
			ParameterSize: model.Details.ParameterSize,
			Quantization:  model.Details.QuantizationLevel,
			Family:        model.Details.Family,
			ExpiresAt:     model.ExpiresAt,
		}

		show := m.getShow(model.Name)
		if show != nil {
			arch := modelInfoString(show.ModelInfo, "general.architecture")
			if arch == "" {
				arch = model.Details.Family
			}

			nLayers := modelInfoInt(show.ModelInfo, arch+".block_count")
			nHeads := modelInfoInt(show.ModelInfo, arch+".attention.head_count")
			nKVHeads := modelInfoInt(show.ModelInfo, arch+".attention.head_count_kv")
			embLen := modelInfoInt(show.ModelInfo, arch+".embedding_length")
			ctxLen := modelInfoInt(show.ModelInfo, arch+".context_length")

			if numCtx := paramInt(show.Parameters, "num_ctx"); numCtx > 0 {
				ctxLen = numCtx
			}
			if ctxLen == 0 {
				ctxLen = 2048
			}
			rm.ContextWindow = ctxLen

			if nLayers > 0 && nKVHeads > 0 && nHeads > 0 && embLen > 0 {
				headDim := embLen / nHeads
				bytesPerElem := kvDtypeBytesPerElement(kvDtype)
				bytesPerToken := int(float64(2*nLayers*nKVHeads*headDim) * bytesPerElem)
				maxBytes := int64(bytesPerToken) * int64(ctxLen)

				rm.KVCache = KVCacheInfo{
					DType:         kvDtype,
					BytesPerToken: bytesPerToken,
					MaxSizeBytes:  maxBytes,
					MaxSizeMiB:    float64(maxBytes) / (1024 * 1024),
				}

				weightsEst := model.SizeVRAM - maxBytes
				if weightsEst < 0 {
					weightsEst = model.Size
				}
				rm.VRAM = VRAMBreakdown{
					TotalBytes:      model.SizeVRAM,
					WeightsEstBytes: weightsEst,
					KVCacheMaxBytes: maxBytes,
				}
			}
		}

		stats.RunningModels = append(stats.RunningModels, rm)
	}

	return stats
}

func (m *OllamaMonitor) getJSON(path string, v interface{}) error {
	resp, err := m.client.Get(m.host + path)
	if err != nil {
		return err
	}
	defer resp.Body.Close()
	return json.NewDecoder(resp.Body).Decode(v)
}

func (m *OllamaMonitor) getShow(name string) *ollamaShowResponse {
	if cached, ok := m.showCache[name]; ok {
		return cached
	}
	body := fmt.Sprintf(`{"model":%q,"verbose":true}`, name)
	resp, err := m.client.Post(m.host+"/api/show", "application/json", strings.NewReader(body))
	if err != nil {
		return nil
	}
	defer resp.Body.Close()

	var show ollamaShowResponse
	if err := json.NewDecoder(resp.Body).Decode(&show); err != nil {
		return nil
	}
	m.showCache[name] = &show
	return &show
}

// Helpers

func modelInfoInt(info map[string]interface{}, key string) int {
	v, ok := info[key]
	if !ok {
		return 0
	}
	switch n := v.(type) {
	case float64:
		return int(n)
	default:
		return 0
	}
}

func modelInfoString(info map[string]interface{}, key string) string {
	v, ok := info[key]
	if !ok {
		return ""
	}
	s, _ := v.(string)
	return s
}

func kvDtypeBytesPerElement(dtype string) float64 {
	switch dtype {
	case "q4_0":
		return 0.5625 // 18 bytes per block of 32
	case "q8_0":
		return 1.0625 // 34 bytes per block of 32
	default: // f16
		return 2.0
	}
}

func paramInt(params string, key string) int {
	for _, line := range strings.Split(params, "\n") {
		parts := strings.Fields(strings.TrimSpace(line))
		if len(parts) == 2 && parts[0] == key {
			return parseInt(parts[1])
		}
	}
	return 0
}
