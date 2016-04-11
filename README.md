# Community Detection

This is an attempt to make a tool for community detection on weighted graphs.

## Features
* Modularity computation
* Louvain clustering on weighted graphs
* JUNG library and GraphML format support

## What's planned
* Features detection

## Requirements
* `Scala v2.11`
* `SBT v0.13`

## Build
    $ sbt package

## Input data format
There are two ways to use this library: either to use a `GraphFactory` class,
or construct a JUNG `Graph[Long, Edge]` yourself.

If you choose to read a `graphml` file with a factory, it should contain a
desctiption of an undirected graph with a `weight` key in its every edge.

