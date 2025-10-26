package com.backend.backend.controller;

import org.springframework.web.bind.annotation.*;
import java.io.*;

@RestController
@RequestMapping("/api")
@CrossOrigin(origins = "*") // allow frontend calls
public class QueryController {

    @PostMapping("/english-to-sql")
    public String englishToSql(@RequestBody String question) {
        String scriptPath = "../rag_pipeline/english_to_sql.py"; // path from backend folder
        return runPythonScript(scriptPath, question);
    }

    private String runPythonScript(String scriptPath, String userInput) {
        StringBuilder output = new StringBuilder();
        try {
            ProcessBuilder pb = new ProcessBuilder("python", scriptPath);
            pb.redirectErrorStream(true);
            Process process = pb.start();

            // Write input to the Python process
            BufferedWriter writer = new BufferedWriter(
                    new OutputStreamWriter(process.getOutputStream()));
            writer.write(userInput);
            writer.newLine();
            writer.flush();
            writer.close();

            // Read output
            BufferedReader reader = new BufferedReader(
                    new InputStreamReader(process.getInputStream()));
            String line;
            while ((line = reader.readLine()) != null) {
                output.append(line).append("\n");
            }

            process.waitFor();
        } catch (Exception e) {
            e.printStackTrace();
            output.append("Error: ").append(e.getMessage());
        }
        return output.toString();
    }
}
