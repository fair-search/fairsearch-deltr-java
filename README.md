# Fair search DELTR for Java

[![image](https://api.travis-ci.org/fair-search/fairsearchcore-java.svg?branch=master)](https://travis-ci.org/fair-search/fairsearchcore-java)
[![image](https://img.shields.io/pypi/l/fairsearchcore.svg)](https://pypi.org/project/fairsearchcore/)

This is the Java library that implements the [DELTR](https://arxiv.org/pdf/1805.08716.pdf) model for fair ranking.

## Installation

You can import the library with maven in your `pom.xml` file:
```xml
<dependency>
  <groupId>com.github.fair-search</groupId>
  <artifactId>fairsearch-deltr</artifactId>
  <version>1.0.0</version>
</dependency>
```
or, if you are using Gradle, in your `build.gradle` file add this in the `dependencies` block:
```gradle
compile "com.github.fair-search:fairsearch-deltr:1.0.0"
```

And, that's it!

## Using it in your code

Add the JAR file to the build path of your project and you are *set*. The key methods are contained in the following class:
- `com.github.fairsearch.Deltr`

The library contains sufficient Java doc for each of the functions.

## Sample usage
Creating and analyzing mtables:

### Train a model

You need to train the model before it can rank documents.
```java
// import other helper libraries
```

### Use the model to rank 

Now, you can use the obtained model to rank some data.
```java
// load some test/prediction data
```

The library contains sufficient code documentation for each of the functions.

### Checking the model a bit deeper

You can check how the training of the model progressed using a special property called `log` (`getLog()`).
```java

```
The `log` returns a list of objects from the `com.github.fairsearch.deltr.models.TrainStep` class. 
The class is a representation of the parameters in each step of the training. The parameters can be acquired with:

- `getTimestamp()`
- `getOmega()`
- `getLoss()` 
- `getLossStandard()`
- `getLossExposure()`

## Development

1. Clone this repository `git clone https://github.com/fair-search/fairsearchdeltr-python`
2. Change directory to the directory where you cloned the repository `cd WHERE_ITS_DOWNLOADED/fairsearchdeltr-python`
3. Use any IDE to work with the code

## Testing

Just run:
```
./gradlew clean check
```
*Note*: Due to the high volume of the datasets, the tests take a bit *longer* time to execute. Also, because there is a *randomness* factor involved in 
the tests, it can happen that (very rarely) they fail sometimes.  

## Credits

The DELTR algorithm is described in this paper:

* Zehlike, Meike, and Carlos Castillo. "[Reducing Disparate Exposure in Ranking:
A Learning to Rank Approach](https://doi.org/10.1145/3132847.3132938)." arXiv preprint arXiv:1805.08716 (2018).

This library was developed by [Ivan Kitanovski](http://ivankitanovski.com/) based on the paper. See the 
[license](https://github.com/fair-search/fairsearchdeltr-java/blob/master/LICENSE) file for more information.

## See also

You can also see the [DELTR plug-in for ElasticSearch](https://github.com/fair-search/fairsearchdeltr-elasticsearch-plugin)
 and [DELTR Python library](https://github.com/fair-search/fairsearchdeltr-python).



 
  
