package com.ambergarden.samples.neo4j;

import org.neo4j.ogm.session.SessionFactory;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.data.neo4j.config.Neo4jConfiguration;
import org.springframework.data.neo4j.repository.config.EnableNeo4jRepositories;
import org.springframework.transaction.annotation.EnableTransactionManagement;

/**
 * Spring JavaConfig configuration class to setup a Spring container and infrastructure components.
 */
@Configuration
@EnableNeo4jRepositories(basePackages = "neo4j.repository")
//Spring data neo4j所需要使用的各个repository存于neo4j.repository中
@EnableTransactionManagement
public class GraphDBConfiguration extends Neo4jConfiguration {
    @Bean
    public org.neo4j.ogm.config.Configuration getConfiguration() {
        org.neo4j.ogm.config.Configuration config =
                new org.neo4j.ogm.config.Configuration();
        // TODO: Temporary uses the embedded driver. We need to switch to http
        // driver. Then we can horizontally scale neo4j
        config.driverConfiguration()
                .setDriverClassName("org.neo4j.ogm.drivers.embedded.driver.EmbeddedDriver")
                .setURI("file:/D:/neo4j-community-3.5.4/data/databases/graph.db/");
        return config;
    }

    @Override
    @Bean
    public SessionFactory getSessionFactory() {
        // Return the session factory which also includes the persistent entities
        return new SessionFactory(getConfiguration(), "neo4j.entity");
        //spring data neo4j所需的各个实体存储于neo.4j和entity中
    }
}
