package com.backend.backend.dto;

public class QueryRequest {
    private String question;
    private boolean execute;
    private int topK = 3;

    public QueryRequest() {}

    public QueryRequest(String question, boolean execute, int topK) {
        this.question = question;
        this.execute = execute;
        this.topK = topK;
    }

    public String getQuestion() {
        return question;
    }

    public void setQuestion(String question) {
        this.question = question;
    }

    public boolean isExecute() {
        return execute;
    }

    public void setExecute(boolean execute) {
        this.execute = execute;
    }

    public int getTopK() {
        return topK;
    }

    public void setTopK(int topK) {
        this.topK = topK;
    }
}
