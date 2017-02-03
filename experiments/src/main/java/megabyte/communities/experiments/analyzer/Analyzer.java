package megabyte.communities.experiments.analyzer;

import com.google.common.base.Function;
import edu.uci.ics.jung.algorithms.cluster.WeakComponentClusterer;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.io.GraphIOException;
import megabyte.communities.entities.Edge;
import megabyte.communities.util.GraphFactory;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.File;
import java.io.IOException;
import java.util.*;
import java.util.stream.Collectors;

public class Analyzer {

    private static final Logger LOG = LoggerFactory.getLogger(Analyzer.class);

    private <T> Map<T, Integer> countSpectrum(List<T> items) {
        Map<T, Integer> spectrum = new TreeMap<>();
        for (T item : items) {
            Integer usersCount = spectrum.get(item);
            if (usersCount == null) {
                usersCount = 1;
            } else {
                usersCount++;
            }
            spectrum.put(item, usersCount);
        }
        return spectrum;
    }

    private void printStats(List<Integer> values) {
        Map<Integer, Integer> spectrum = countSpectrum(values);
        System.out.println("  Spectrum:");
        for (Map.Entry<Integer, Integer> entry : spectrum.entrySet()) {
            int value = entry.getKey();
            int count = entry.getValue();
            System.out.println("    " + value + " -> " + count);
        }
        System.out.println("  Average: " + values.stream().mapToInt(i -> i).average().getAsDouble());
    }

    private List<Integer> findNetworksSizes(Graph<String, Edge> graph) {
        Function<Graph<String, Edge>, Set<Set<String>>> transformer = new WeakComponentClusterer<>();
        Set<Set<String>> clusters = transformer.apply(graph);
        List<Integer> networksSizes = new ArrayList<>();
        for (Set<String> cluster : clusters) {
            for (int i = 0; i < cluster.size(); i++) {
                networksSizes.add(cluster.size());
            }
        }
        return networksSizes;
    }

    private void doMain(String graphFile) throws IOException, GraphIOException {
        //GraphFactory graphFactory = GraphFactory.getInstance();
        Graph<String, Edge> userGraph = GraphFactory.readGraph(new File(graphFile));

        LOG.info("Analyzing the social graph");
        System.out.println("Users: " + userGraph.getVertexCount());
        List<Integer> friendsCounts = userGraph.getVertices().stream()
                .map(userId -> userGraph.getOutEdges(userId).size()).collect(Collectors.toList());
        System.out.println("Edges: " + friendsCounts.stream().mapToInt(i -> i).sum());

        System.out.println("Friends:");
        printStats(friendsCounts);


        List<Integer> networksSizes = findNetworksSizes(userGraph);
        System.out.println("Network size:");
        printStats(networksSizes);
    }

    public static void main(String... args) throws IOException, GraphIOException {
        new Analyzer().doMain(args[0]);
    }
}
