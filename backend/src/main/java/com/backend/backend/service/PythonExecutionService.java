package com.backend.backend.service;


import com.backend.backend.dto.QueryResponse;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;

import java.io.BufferedReader;
import java.io.InputStreamReader;
import java.nio.file.Paths;
import java.util.ArrayList;
import java.util.List;

@Service
public class PythonExecutionService {

    @Value("${python.script.path:../rag_pipeline}")
    private String pythonScriptPath;

    @Value("${python.executable:python3}")
    private String pythonExecutable;

    public QueryResponse executeQuery(String question, boolean execute, int topK) throws Exception {
        List<String> command = new ArrayList<>();
        command.add(pythonExecutable);
        command.add(Paths.get(pythonScriptPath, "generate_sql_from_query.py").toString());

        if (execute) {
            command.add("--execute");
        }
        command.add("--top-k");
        command.add(String.valueOf(topK));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);

        Process process = pb.start();

        // Write question to stdin
        process.getOutputStream().write((question + "\n").getBytes());
        process.getOutputStream().flush();
        process.getOutputStream().close();

        // Read output
        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }

        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new RuntimeException("Python script failed with exit code: " + exitCode + "\nOutput: " + output.toString());
        }

        return parseOutput(output.toString(), execute);
    }

    public void buildSchema() throws Exception {
        List<String> command = new ArrayList<>();
        command.add(pythonExecutable);
        command.add(Paths.get(pythonScriptPath, "build_schema.py").toString());

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);

        Process process = pb.start();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }

        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new RuntimeException("Schema build failed: " + output.toString());
        }
    }

    public String retrieveSchema(String question, int topK) throws Exception {
        List<String> command = new ArrayList<>();
        command.add(pythonExecutable);
        command.add(Paths.get(pythonScriptPath, "retrieve_schema.py").toString());
        command.add("--top-k");
        command.add(String.valueOf(topK));

        ProcessBuilder pb = new ProcessBuilder(command);
        pb.redirectErrorStream(true);

        Process process = pb.start();

        process.getOutputStream().write((question + "\n").getBytes());
        process.getOutputStream().flush();
        process.getOutputStream().close();

        BufferedReader reader = new BufferedReader(new InputStreamReader(process.getInputStream()));
        StringBuilder output = new StringBuilder();
        String line;
        while ((line = reader.readLine()) != null) {
            output.append(line).append("\n");
        }

        int exitCode = process.waitFor();

        if (exitCode != 0) {
            throw new RuntimeException("Schema retrieval failed: " + output.toString());
        }

        return output.toString();
    }

    private QueryResponse parseOutput(String output, boolean wasExecuted) {
        QueryResponse response = new QueryResponse();
        response.setSuccess(true);

        String[] sections = output.split("--- ");

        for (String section : sections) {
            section = section.trim();

            if (section.startsWith("Generated SQL")) {
                String sql = extractSqlFromSection(section);
                response.setSql(sql);
            } else if (section.startsWith("Results") && wasExecuted) {
                response.setExecuted(true);
                String resultsData = extractResultsFromSection(section);
                response.setResultsText(resultsData);
            }
        }

        if (response.getSql() == null || response.getSql().isEmpty()) {
            response.setSql(extractSqlFallback(output));
        }

        return response;
    }

    private String extractSqlFromSection(String section) {
        String[] lines = section.split("\n");
        StringBuilder sql = new StringBuilder();

        boolean foundSqlMarker = false;
        for (String line : lines) {
            if (line.trim().equals("Generated SQL ---")) {
                foundSqlMarker = true;
                continue;
            }
            if (foundSqlMarker) {
                if (line.contains("--- Results") || line.contains("---")) {
                    break;
                }
                sql.append(line.trim()).append("\n");
            }
        }

        return sql.toString().trim();
    }

    private String extractResultsFromSection(String section) {
        String[] lines = section.split("\n");
        StringBuilder results = new StringBuilder();

        boolean foundResultsMarker = false;
        for (String line : lines) {
            if (line.trim().equals("Results ---")) {
                foundResultsMarker = true;
                continue;
            }
            if (foundResultsMarker) {
                results.append(line).append("\n");
            }
        }

        return results.toString().trim();
    }

    private String extractSqlFallback(String output) {
        String[] lines = output.split("\n");
        StringBuilder sql = new StringBuilder();

        boolean inSqlBlock = false;
        for (String line : lines) {
            String trimmed = line.trim();
            if (trimmed.toUpperCase().startsWith("SELECT")) {
                inSqlBlock = true;
            }
            if (inSqlBlock) {
                sql.append(trimmed).append("\n");
                if (trimmed.endsWith(";")) {
                    break;
                }
            }
        }

        return sql.toString().trim();
    }
}