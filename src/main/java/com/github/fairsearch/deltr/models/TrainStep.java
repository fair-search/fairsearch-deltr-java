package com.github.fairsearch.deltr.models;

import org.nd4j.linalg.api.ndarray.INDArray;

public class TrainStep {

    private long timestamp;
    private INDArray omega;
    private INDArray cost;
    private INDArray grad;
    private double lossStandard;
    private double lossExposure;
    private double totalCost;

    public double getTotalCost() {
        return totalCost;
    }

    public void setTotalCost(double totalCost) {
        this.totalCost = totalCost;
    }

    public TrainStep(long timestamp, INDArray omega, INDArray cost, INDArray grad, double lossStandard, double lossExposure) {
        this.timestamp = timestamp;
        this.omega = omega;
        this.cost = cost;
        this.grad = grad;
        this.lossStandard = lossStandard;
        this.lossExposure = lossExposure;
    }

    public TrainStep(long timestamp, INDArray cost, double lossStandard, double lossExposure) {
        this.timestamp = timestamp;
        this.cost = cost;
        this.lossStandard = lossStandard;
        this.lossExposure = lossExposure;
    }

    public long getTimestamp() {
        return timestamp;
    }

    public void setTimestamp(long timestamp) {
        this.timestamp = timestamp;
    }

    public INDArray getOmega() {
        return omega;
    }

    public void setOmega(INDArray omega) {
        this.omega = omega;
    }

    public INDArray getCost() {
        return cost;
    }

    public void setCost(INDArray cost) {
        this.cost = cost;
    }

    public INDArray getGrad() {
        return grad;
    }

    public void setGrad(INDArray grad) {
        this.grad = grad;
    }

    public double getLossStandard() {
        return lossStandard;
    }

    public void setLossStandard(double lossStandard) {
        this.lossStandard = lossStandard;
    }

    public double getLossExposure() {
        return lossExposure;
    }

    public void setLossExposure(double lossExposure) {
        this.lossExposure = lossExposure;
    }
}
