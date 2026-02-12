:Section
--------
doc_id         : STRING    -- foreign key to :Document.id (your stable doc id)
section_index  : INTEGER   -- ordering within the document (e.g., 8)
title          : STRING    -- section title (e.g., "DND employees and CAF members")

Notes
-----
- Uniqueness is typically (doc_id, section_index) or (doc_id, title) depending on your loader.
- Neo4j Browser shows <elementId> and <id> which are internal, not your schema.
*/