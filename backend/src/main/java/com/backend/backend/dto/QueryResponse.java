package com.backend.backend.dto;

public class QueryResponse {
    private boolean success;
    private String sql;
    private boolean executed;
    private String error;
    private String resultsText;

    public QueryResponse() {}

    public boolean isSuccess() {
        return success;
    }

    public void setSuccess(boolean success) {
        this.success = success;
    }

    public String getSql() {
        return sql;
    }

    public void setSql(String sql) {
        this.sql = sql;
    }

    public boolean isExecuted() {
        return executed;
    }

    public void setExecuted(boolean executed) {
        this.executed = executed;
    }

    public String getError() {
        return error;
    }

    public void setError(String error) {
        this.error = error;
    }

    public String getResultsText() {
        return resultsText;
    }

    public void setResultsText(String resultsText) {
        this.resultsText = resultsText;
    }
}