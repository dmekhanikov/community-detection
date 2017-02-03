package megabyte.communities.experiments.transformer;

import edu.uci.ics.jung.graph.DirectedSparseGraph;
import edu.uci.ics.jung.graph.Graph;
import edu.uci.ics.jung.io.GraphMLWriter;
import megabyte.communities.experiments.dao.MongoDAO;
import org.apache.commons.csv.CSVFormat;
import org.apache.commons.csv.CSVParser;
import org.apache.commons.csv.CSVRecord;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.io.*;
import java.util.*;
import java.util.stream.Collectors;

public class GraphTransformer {

    private static final Logger LOG = LoggerFactory.getLogger(GraphTransformer.class);

    private static final String CITY = "Singapore";
    private static final String BASE_DIR = "experiments/src/main/resources/" + CITY;
    private static final String IDS_DIR = BASE_DIR + "/ids";
    private static final String OUT_DIR = BASE_DIR + "/graphs";

    private static final String TWITTER = "twitter";
    private static final String INSTAGRAM = "instagram";
    private static final String FOURSQUARE = "foursquare";
    private static final String[] NETWORKS = {TWITTER, INSTAGRAM, FOURSQUARE};

    private void transform() throws IOException {
        List<Set<User>> userSets = new ArrayList<>();
        for (String network : NETWORKS) {
            Set<User> users = mergeCopies(readGraph(network));
            userSets.add(users);
        }
        Set<Long> intersection = findIntersection(userSets);
        LOG.info("Result user set size: " + intersection.size());

        for (int i = 0; i < NETWORKS.length; i++) {
            Set<User> users = userSets.get(i).stream()
                    .filter(user -> intersection.contains(user.getTwitterId()))
                    .collect(Collectors.toSet());
            Graph<Long, String> jungGraph = toJungGraph(users);
            File outFile = new File(OUT_DIR, NETWORKS[i] + ".graphml");
            outFile.getParentFile().mkdirs();
            write(jungGraph, outFile);
        }
    }

    private Collection<User> readGraph(String network) throws IOException {
        boolean isTwitter = TWITTER.equals(network);
        File idsFile = new File(IDS_DIR, network + ".csv");
        Map<Long, User> graph = readUsers(idsFile, isTwitter);
        readEdges(graph, network);
        return graph.values();
    }

    private Map<Long, User> readUsers(File inputFile, boolean isTwitter) throws IOException {
        LOG.info("Reading users from file " + inputFile);
        Reader reader = new BufferedReader(new FileReader(inputFile));
        Map<Long, User> users = new HashMap<>();
        try (CSVParser parser = new CSVParser(reader, CSVFormat.DEFAULT.withFirstRecordAsHeader())) {
            for (CSVRecord record : parser) {
                long id = Long.parseLong(record.get(0));
                User user;
                if (isTwitter) {
                    user = new User(id);
                } else {
                    long twitterId = Long.parseLong(record.get(1));
                    user = new User(twitterId);
                }
                users.put(id, user);
            }
        }
        LOG.info(users.size() + " users read");
        return users;
    }

    private void readEdges(Map<Long, User> users, String collection) {
        LOG.info("Getting friends from " + collection);
        MongoDAO dao = new MongoDAO(MongoDAO.CRAWLER_DB);
        for (Map.Entry<Long, User> entry : users.entrySet()) {
            long userId = entry.getKey();
            User user = entry.getValue();
            List<Long> friends = dao.getFriends(collection, userId);
            if (friends != null) {
                for (long friendId : friends) {
                    User friend = users.get(friendId);
                    if (friend != null) {
                        user.addEdge(friend);
                    }
                }
            }
        }
    }

    private Set<Long> findIntersection(List<Set<User>> userSets) {
        Set<Long> intersection = null;
        for (Set<User> users : userSets) {
            Set<Long> userIds = users.stream()
                    .map(User::getTwitterId)
                    .collect(Collectors.toSet());
            if (intersection == null) {
                intersection = new HashSet<>(userIds);
            } else {
                intersection.retainAll(userIds);
            }
        }
        assert intersection != null;
        Set<Long> referenced = getReferenced(userSets);
        intersection.retainAll(referenced);
        return intersection;
    }

    private Set<Long> getReferenced(List<Set<User>> userSets) {
        Set<Long> referenced = new HashSet<>();
        for (Set<User> users : userSets) {
            for (User user : users) {
                if (!user.getEdges().isEmpty()) {
                    referenced.add(user.getTwitterId());
                    for (User e : user.getEdges()) {
                        referenced.add(e.getTwitterId());
                    }
                }
            }
        }
        return referenced;
    }

    private Set<User> mergeCopies(Collection<User> users) {
        Map<Long, User> merged = new HashMap<>();
        for (User user : users) {
            User storedUser = merged.get(user.getTwitterId());
            if (storedUser != null) {
                List<User> edgesDiff = user.getEdges().stream()
                        .filter(storedUser.getEdges()::contains)
                        .collect(Collectors.toList());
                storedUser.getEdges().addAll(edgesDiff);
            } else {
                merged.put(user.getTwitterId(), user);
            }
        }
        return new HashSet<>(merged.values());
    }

    private Graph<Long, String> toJungGraph(Set<User> users) {
        Graph<Long, String> graph = new DirectedSparseGraph<>();
        for (User fromUser : users) {
            long fromId = fromUser.getTwitterId();
            for (User toUser : fromUser.getEdges()) {
                if (users.contains(toUser)) {
                    long toId = toUser.getTwitterId();
                    String edgeName = fromId + "->" + toId;
                    graph.addEdge(edgeName, fromId, toId);
                }
            }
        }
        return graph;
    }

    private <V, E> void write(Graph<V, E> graph, File outFile) throws IOException {
        try (Writer writer = new BufferedWriter(new FileWriter(outFile))) {
            GraphMLWriter<V, E> graphWriter = new GraphMLWriter<>();
            graphWriter.addEdgeData("weight", null, "1.0", edge -> null);
            graphWriter.save(graph, writer);
        }
    }

    public static void main(String... args) throws IOException {
        GraphTransformer transformer = new GraphTransformer();
        transformer.transform();
    }
}
