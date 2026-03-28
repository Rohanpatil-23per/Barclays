def check_if_attack(incoming_768d_vector: np.ndarray, distance_threshold: float = 0.35):
    """
    Checks if the incoming vector matches any previously confirmed attacks.
    Lower distance = Higher similarity.
    """
    with conn.cursor() as cur:
        # Find the distance to the single closest confirmed attack in history
        cur.execute("""
            SELECT (layer1_768d <-> %s) AS distance 
            FROM threat_memory 
            ORDER BY distance ASC 
            LIMIT 1;
        """, (incoming_768d_vector,))
        
        result = cur.fetchone()
        
        # If the database is empty (very first run), assume it's normal
        if result is None:
            return False, float('inf')
            
        closest_distance = result[0]
        
        # If the vector is extremely similar to a past attack, flag it!
        if closest_distance < distance_threshold:
            print(f"⚠️ MATCH FOUND! Distance: {closest_distance:.4f} (Threshold: {distance_threshold})")
            return True, closest_distance
        else:
            print(f"✅ NORMAL. Distance to nearest attack: {closest_distance:.4f}")
            return False, closest_distance