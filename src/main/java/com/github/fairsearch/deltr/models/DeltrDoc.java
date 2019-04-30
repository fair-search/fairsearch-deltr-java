package com.github.fairsearch.deltr.models;

import java.util.List;

/**
 * This is an interface describing the methods a document must implement for DELTR to operate
 */
public interface DeltrDoc {

    /**
     * Unique ID of the document within the ranking
     * @return      The ID
     */
    int id();

    /**
     * Returns the number of features in the document (excluding the judgement
     * @return      The length of the feature list/map
     */
    int size();

    /**
     * Returns a boolean indicating whether the element is protected or not
     * @return      Boolean indicating if it is protected or not
     */
    boolean isProtected();

    /**
     * Returns the judgement of this document. Ideally, that should not be part of the assignFeature matrix
     * @return      The judgment
     */
    double judgement();

    /**
     * Sets a new judgement for this document.
     * @param judgement     The new judgement value
     */
    void rejudge(double judgement);

    /**
     * List of feature names/keys of the document
     * @return      A list of feature names
     */
    List<String> keys();

    /**
     * List of feature values of the document
     * @return      A list of feature of values
     */
    List<Double> features();

    /**
     * Returns the assignFeature at position `index`
     * @param index     The position of the feature to return
     * @return          The value of the feature
     */
    Double feature(int index);

    /**
     * Returns the assignFeature with the name `name`
     * @param name      The name of the feature to return
     * @return          The value of the feature
     */
    Double feature(String name);

    /**
     * Set `value` of the feature with name `name`
     * @param name      The name of the feature to set
     * @param value     The value of the feature to set
     */
    void put(String name, Double value);

    /**
     * Set `value` of the protected feature with name `name`. There can be only *one* such feature.
     * @param name          The name of the feature to set
     * @param isProtected   A boolean indicating if the element is protected or not
     */
    void put(String name, Boolean isProtected);

    /**
     * Returns the `name` of the feature that stores the boolean indicating if the element is protected
     * @return              The name of of the protected feature
     */
    String protectedFeatureName();

    /**
     * Returns the `index` of the feature that stores the boolean indicating if the element is protected in the feature list
     * @return              The position of of the protected feature in the feature list
     */
    int protectedFeatureIndex();
}
