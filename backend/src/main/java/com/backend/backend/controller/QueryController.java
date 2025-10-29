package com.backend.backend.controller;


import com.backend.backend.dto.QueryRequest;
import com.backend.backend.dto.QueryResponse;
import com.backend.backend.service.PythonExecutionService;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*")
public class QueryController {

    @Autowired
    private PythonExecutionService pythonExecutorService;

    @PostMapping("/query")
    public ResponseEntity<QueryResponse> executeQuery(@RequestBody QueryRequest request) {
        try {
            QueryResponse response = pythonExecutorService.executeQuery(
                    request.getQuestion(),
                    request.isExecute(),
                    request.getTopK()
            );
            return ResponseEntity.ok(response);
        } catch (Exception e) {
            QueryResponse errorResponse = new QueryResponse();
            errorResponse.setSuccess(false);
            errorResponse.setError("Error: " + e.getMessage());
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR).body(errorResponse);
        }
    }

    @GetMapping("/health")
    public ResponseEntity<String> healthCheck() {
        return ResponseEntity.ok("Backend is running successfully!");
    }

    @PostMapping("/build-schema")
    public ResponseEntity<String> buildSchema() {
        try {
            pythonExecutorService.buildSchema();
            return ResponseEntity.ok("Schema built successfully!");
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error building schema: " + e.getMessage());
        }
    }

    @PostMapping("/retrieve-schema")
    public ResponseEntity<String> retrieveSchema(@RequestBody QueryRequest request) {
        try {
            String result = pythonExecutorService.retrieveSchema(request.getQuestion(), request.getTopK());
            return ResponseEntity.ok(result);
        } catch (Exception e) {
            return ResponseEntity.status(HttpStatus.INTERNAL_SERVER_ERROR)
                    .body("Error retrieving schema: " + e.getMessage());
        }
    }
}
