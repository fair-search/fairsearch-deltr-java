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
     * Returns the judgement of this document. Ideally, that should not be part of the assignFeature matrix
     * @return
     */
    double judgement();

    /**
     * Sets a new judgement for this document.
     * @return
     */
    void rejudge(double judgement);

    /**
     * List of feature names/keys of the document
     * @return
     */
    List<String> keys();

    /**
     * List of feature values of the document
     * @return
     */
    List<Double> features();

    /**
     * Returns the assignFeature at position `index`
     * @param index
     * @return
     */
    Double feature(int index);

    /**
     * Returns the assignFeature with the name `name`
     * @param name
     * @return
     */
    Double feature(String name);

    /**
     * Set `value` of the feature with name `name`
     * @param name
     * @return
     */
    void set(String name, Double value);


    /**
     * Returns the `name` of the protected assignFeature
     * @return
     */
    String protectedFeatureName();

    /**
     * Returns the `index` of the protected assignFeature in the assignFeature list
     * @return
     */
    int protectedFeatureIndex();
}
