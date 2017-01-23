package megabyte.communities.transformer;

import java.util.ArrayList;
import java.util.Collection;
import java.util.List;

public class User {

    private final long twitterId;

    private final List<User> edges = new ArrayList<>();

    public User(long twitterId) {
        this.twitterId = twitterId;
    }

    public long getTwitterId() {
        return twitterId;
    }

    public Collection<User> getEdges() {
        return edges;
    }

    public void addEdge(User to) {
        edges.add(to);
    }

    @Override
    public boolean equals(Object o) {
        if (this == o) return true;
        if (o == null || getClass() != o.getClass()) return false;

        User user = (User) o;

        return twitterId == user.twitterId;
    }

    @Override
    public int hashCode() {
        return (int) (twitterId ^ (twitterId >>> 32));
    }
}
