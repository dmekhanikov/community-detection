# Community Detection

This is an attempt to make a community detection tool.

## Features
* Modularity computation
* Louvain and Spectral clustering
* JUNG library and GraphML format support

## What's planned
* Multilayer constrained clustering

## Requirements
* `Scala v2.11`
* `SBT v0.13`

## Build
    $ sbt package

## Input data format
There are two ways to use this library: either to use a `GraphFactory` class,
or construct a JUNG `Graph[Long, Edge]` yourself.

If you choose to read a `graphml` file with a factory, it should contain a
description of an undirected graph. Edges may contain a `weight` key. 
Default value of the weight is 1.0.

