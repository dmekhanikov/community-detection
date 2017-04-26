package megabyte.communities.experiments.dao;

import com.mongodb.MongoClient;
import com.mongodb.client.FindIterable;
import com.mongodb.client.MongoCollection;
import com.mongodb.client.MongoDatabase;
import com.mongodb.client.model.Filters;
import org.bson.Document;
import org.slf4j.Logger;
import org.slf4j.LoggerFactory;

import java.util.ArrayList;
import java.util.Iterator;
import java.util.List;

public class MongoDAO {

    private static final Logger LOG = LoggerFactory.getLogger(MongoDAO.class);

    private static final String USER_ID = "userId";
    private static final String FRIENDS = "friends";

    public static final String CRAWLER_DB = "crawler";
    private static final String HOST = "localhost";
    private static final int PORT = 27017;
    private final MongoDatabase db;

    public MongoDAO(String database) {
        MongoClient mongoClient = new MongoClient(HOST, PORT);
        this.db = mongoClient.getDatabase(database);
        LOG.info("Database connection initialized");
    }

    public List<Long> getFriends(String collectionName, long id) {
        MongoCollection<Document> collection = db.getCollection(collectionName);
        FindIterable<Document> findResult = collection.find(Filters.eq(USER_ID, id));
        Iterator<Document> it = findResult.iterator();
        if (!it.hasNext()) {
            return null;
        }
        Document user = it.next();
        if (it.hasNext()) {
            LOG.warn("Several entries for user " + id + " found");
            return null;
        }
        return (List<Long>) user.get(FRIENDS);
    }

    public List<Long> getUserIds(String collectionName) {
        MongoCollection<Document> collection = db.getCollection(collectionName);
        FindIterable<Document> findResult = collection.find();

        List<Long> ids = new ArrayList<>();
        for (Document user : findResult) {
            ids.add(user.getLong("userId"));
        }
        return ids;
    }
}
