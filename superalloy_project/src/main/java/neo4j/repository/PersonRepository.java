package neo4j.repository;

import neo4j.entity.Person;
import org.springframework.data.neo4j.repository.GraphRepository;

public interface PersonRepository extends GraphRepository<Person>{

}
