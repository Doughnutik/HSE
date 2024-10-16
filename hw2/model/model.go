package model

type DecodeRequest struct {
	InputString string `json:"inputString"`
}

type DecodeResponse struct {
	OutputString string `json:"outputString"`
}

const Port int = 4015
