package main

import (
	"fmt"
	"os/exec"
	"strconv"
	"strings"
	"sync"
	"time"
)

type GPUProcess struct {
	PID         int    `json:"pid"`
	ProcessName string `json:"process_name"`
	UsedMemory  int    `json:"used_memory_mib"`
}

type GPUInfo struct {
	Index             int          `json:"index"`
	Name              string       `json:"name"`
	UUID              string       `json:"uuid"`
	DriverVersion     string       `json:"driver_version"`
	TemperatureC      int          `json:"temperature_c"`
	FanSpeedPct       int          `json:"fan_speed_pct"`
	PowerDrawW        float64      `json:"power_draw_w"`
	PowerLimitW       float64      `json:"power_limit_w"`
	MemoryUsedMiB     int          `json:"memory_used_mib"`
	MemoryTotalMiB    int          `json:"memory_total_mib"`
	MemoryFreeMiB     int          `json:"memory_free_mib"`
	GPUUtilizationPct int          `json:"gpu_utilization_pct"`
	MemUtilizationPct int          `json:"mem_utilization_pct"`
	PState            string       `json:"pstate"`
	PCIEGenCurrent    int          `json:"pcie_gen_current"`
	PCIEGenMax        int          `json:"pcie_gen_max"`
	Processes         []GPUProcess `json:"processes"`
}

type GPUMetrics struct {
	Timestamp string    `json:"timestamp"`
	GPUs      []GPUInfo `json:"gpus"`
}

type GPUMonitor struct {
	mu      sync.RWMutex
	latest  *GPUMetrics
	stopCh  chan struct{}
}

func NewGPUMonitor() *GPUMonitor {
	return &GPUMonitor{
		stopCh: make(chan struct{}),
	}
}

func (m *GPUMonitor) Start() {
	m.poll()
	go func() {
		ticker := time.NewTicker(1 * time.Second)
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

func (m *GPUMonitor) Stop() {
	close(m.stopCh)
}

func (m *GPUMonitor) Latest() *GPUMetrics {
	m.mu.RLock()
	defer m.mu.RUnlock()
	return m.latest
}

func (m *GPUMonitor) poll() {
	metrics, err := fetchGPUMetrics()
	if err != nil {
		fmt.Println("nvidia-smi error:", err)
		return
	}
	m.mu.Lock()
	m.latest = metrics
	m.mu.Unlock()
}

func fetchGPUMetrics() (*GPUMetrics, error) {
	gpus, err := queryGPUs()
	if err != nil {
		return nil, err
	}

	procs, err := queryProcesses()
	if err != nil {
		return nil, err
	}

	// Attach processes to GPUs by UUID
	procMap := make(map[string][]GPUProcess)
	for _, p := range procs {
		procMap[p.uuid] = append(procMap[p.uuid], p.proc)
	}
	for i := range gpus {
		if ps, ok := procMap[gpus[i].UUID]; ok {
			gpus[i].Processes = ps
		} else {
			gpus[i].Processes = []GPUProcess{}
		}
	}

	return &GPUMetrics{
		Timestamp: time.Now().UTC().Format(time.RFC3339),
		GPUs:      gpus,
	}, nil
}

type procWithUUID struct {
	uuid string
	proc GPUProcess
}

func queryGPUs() ([]GPUInfo, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-gpu=index,name,uuid,driver_version,temperature.gpu,fan.speed,power.draw,power.limit,memory.used,memory.total,memory.free,utilization.gpu,utilization.memory,pstate,pcie.link.gen.current,pcie.link.gen.max",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return nil, fmt.Errorf("query-gpu: %w", err)
	}

	var gpus []GPUInfo
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ", ")
		if len(fields) < 16 {
			continue
		}
		gpus = append(gpus, GPUInfo{
			Index:             parseInt(fields[0]),
			Name:              fields[1],
			UUID:              fields[2],
			DriverVersion:     fields[3],
			TemperatureC:      parseInt(fields[4]),
			FanSpeedPct:       parseInt(fields[5]),
			PowerDrawW:        parseFloat(fields[6]),
			PowerLimitW:       parseFloat(fields[7]),
			MemoryUsedMiB:     parseInt(fields[8]),
			MemoryTotalMiB:    parseInt(fields[9]),
			MemoryFreeMiB:     parseInt(fields[10]),
			GPUUtilizationPct: parseInt(fields[11]),
			MemUtilizationPct: parseInt(fields[12]),
			PState:            fields[13],
			PCIEGenCurrent:    parseInt(fields[14]),
			PCIEGenMax:        parseInt(fields[15]),
		})
	}
	return gpus, nil
}

func queryProcesses() ([]procWithUUID, error) {
	out, err := exec.Command("nvidia-smi",
		"--query-compute-apps=gpu_uuid,pid,process_name,used_memory",
		"--format=csv,noheader,nounits",
	).Output()
	if err != nil {
		return nil, fmt.Errorf("query-compute-apps: %w", err)
	}

	var procs []procWithUUID
	for _, line := range strings.Split(strings.TrimSpace(string(out)), "\n") {
		line = strings.TrimSpace(line)
		if line == "" {
			continue
		}
		fields := strings.Split(line, ", ")
		if len(fields) < 4 {
			continue
		}
		procs = append(procs, procWithUUID{
			uuid: fields[0],
			proc: GPUProcess{
				PID:         parseInt(fields[1]),
				ProcessName: fields[2],
				UsedMemory:  parseInt(fields[3]),
			},
		})
	}
	return procs, nil
}

func parseInt(s string) int {
	s = strings.TrimSpace(s)
	if s == "[N/A]" || s == "N/A" || s == "" {
		return 0
	}
	v, _ := strconv.Atoi(s)
	return v
}

func parseFloat(s string) float64 {
	s = strings.TrimSpace(s)
	if s == "[N/A]" || s == "N/A" || s == "" {
		return 0
	}
	v, _ := strconv.ParseFloat(s, 64)
	return v
}
