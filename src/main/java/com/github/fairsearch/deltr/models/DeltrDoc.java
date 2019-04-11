package com.github.fairsearch.deltr.models;

import java.util.List;

/**
 * This is an interface describing the methods a document must implement for DELTR to operate
 */
public interface DeltrDoc {

    /**
     * Unique ID of the document within the ranking
     * @return
     */
    int id();

    /**
     * return the number of features in the document (excluding the judgement
     * @return
     */
    int size();

    /**
     * Returns a boolean indicating whether the element is protected or not
     * @return
     */
    boolean isProtected();

    /**
     * Returns the judgement of this document. Ideally, that should not be part of the feature matrix
     * @return
     */
    double judgement();

    /**
     * Sets a new judgement for this document.
     * @return
     */
    void rejudge(double judgement);

    /**
     * List of features of the document
     * @return
     */
    List<Double> features();

    /**
     * Returns the feature at position `index`
     * @param index
     * @return
     */
    Double feature(int index);

    /**
     * Return the feature with named `name`
     * @param name
     * @return
     */
    Double feature(String name);
}
