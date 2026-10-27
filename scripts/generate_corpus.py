"""One-off generator for the bundled demo corpus.

This script authors the ~120 short synthetic knowledge-base articles that ship
with the package at ``src/semantic_search_engine/data/corpus.json`` and writes
them out as a single JSON file. It is not part of the runtime search pipeline
and does not need to be run again unless the corpus itself is being edited.
"""

from __future__ import annotations

import json
from pathlib import Path

# Each entry: (short_id, topic, title, body_text)
# The id prefix encodes the topic so related documents can be spotted at a
# glance in evaluation output (e.g. "astro-07", "ml-12").
DOCUMENTS: list[tuple[str, str, str, str]] = [
    # ---------------------------------------------------------------- Astronomy
    ("astro-01", "astronomy", "How the Solar System Formed",
     "The Solar System formed about 4.6 billion years ago from the gravitational "
     "collapse of a giant molecular cloud of gas and dust. Most of the material "
     "gathered at the center to form the Sun, while the remaining disk of debris "
     "flattened and clumped together to form planets, moons, and asteroids. "
     "This process, called accretion, explains why most planets orbit in roughly "
     "the same plane."),
    ("astro-02", "astronomy", "The Terrestrial Planets",
     "Mercury, Venus, Earth, and Mars are known as the terrestrial planets because "
     "they have solid, rocky surfaces unlike the gas giants farther out. Mercury is "
     "the smallest and closest to the Sun, with extreme temperature swings and no "
     "real atmosphere. Venus has a thick carbon dioxide atmosphere that traps heat "
     "through a runaway greenhouse effect, making it the hottest planet. Mars has a "
     "thin atmosphere, polar ice caps, and the largest volcano in the Solar System, "
     "Olympus Mons."),
    ("astro-03", "astronomy", "Jupiter and Saturn, the Gas Giants",
     "Jupiter and Saturn are gas giants composed mainly of hydrogen and helium with "
     "no solid surface to land on. Jupiter is the largest planet and hosts the Great "
     "Red Spot, a storm that has raged for centuries, along with dozens of moons "
     "including Europa and Io. Saturn is famous for its extensive ring system made "
     "of ice and rock particles, and its moon Titan has a thick atmosphere and "
     "liquid methane lakes."),
    ("astro-04", "astronomy", "Uranus and Neptune, the Ice Giants",
     "Uranus and Neptune are called ice giants because, beyond hydrogen and helium, "
     "they contain large amounts of water, ammonia, and methane ices in their "
     "interiors. Uranus is unusual in that it rotates almost on its side, likely "
     "the result of an ancient collision. Neptune, the most distant known planet, "
     "has the fastest winds recorded in the Solar System, reaching over 2,000 "
     "kilometers per hour."),
    ("astro-05", "astronomy", "The Moon and Ocean Tides",
     "Earth's Moon formed roughly 4.5 billion years ago, most likely from debris "
     "left after a Mars-sized body collided with the young Earth. The Moon's "
     "gravitational pull is the primary driver of ocean tides, creating two high "
     "tides and two low tides at most coastal locations each day. The Sun also "
     "contributes to tides, and when the Sun and Moon align during full and new "
     "moons, the result is a stronger spring tide."),
    ("astro-06", "astronomy", "The Asteroid Belt",
     "The asteroid belt lies between the orbits of Mars and Jupiter and contains "
     "millions of rocky bodies left over from the Solar System's formation. Most "
     "asteroids are too small and scattered to have combined into a planet because "
     "Jupiter's gravity repeatedly disturbed their orbits. Ceres, the largest object "
     "in the belt, is massive enough to be classified as a dwarf planet."),
    ("astro-07", "astronomy", "Comets and the Oort Cloud",
     "Comets are icy bodies that develop glowing comas and tails when their orbits "
     "bring them close to the Sun, causing surface ice to vaporize. Short-period "
     "comets originate from the Kuiper Belt beyond Neptune, while long-period "
     "comets are thought to come from the distant, spherical Oort Cloud that may "
     "extend halfway to the nearest star. Each pass near the Sun erodes a comet "
     "slightly, so most comets eventually break apart or burn out."),
    ("astro-08", "astronomy", "Stellar Classification",
     "Stars are classified by spectral type, letters O, B, A, F, G, K, and M, "
     "ranked from hottest and bluest to coolest and reddest. The Sun is a "
     "G-type main-sequence star, meaning it fuses hydrogen into helium in a "
     "stable core. A star's mass largely determines its temperature, color, "
     "luminosity, and how long it will live, with the most massive stars burning "
     "through their fuel fastest."),
    ("astro-09", "astronomy", "The Life Cycle of Stars",
     "Stars are born in dense clouds of gas called nebulae, where gravity pulls "
     "material together until nuclear fusion ignites. For most of their lives, "
     "stars sit on the main sequence, steadily fusing hydrogen. When a massive "
     "star exhausts its fuel, it can collapse and explode as a supernova, "
     "scattering heavy elements into space and sometimes leaving behind a neutron "
     "star or black hole; smaller stars like the Sun instead swell into red giants "
     "and shed their outer layers as planetary nebulae."),
    ("astro-10", "astronomy", "Black Holes",
     "A black hole is a region of space where gravity is so strong that nothing, "
     "not even light, can escape once past its boundary, the event horizon. Most "
     "black holes form from the collapse of massive stars, though supermassive "
     "black holes millions of times the Sun's mass sit at the centers of most "
     "galaxies, including the Milky Way. Scientists detect black holes indirectly, "
     "through their gravitational pull on nearby stars or by imaging the glowing "
     "gas that swirls around them."),
    ("astro-11", "astronomy", "The Milky Way Galaxy",
     "The Milky Way is a barred spiral galaxy containing an estimated 100 to 400 "
     "billion stars, including our Sun, which sits in one of its spiral arms about "
     "27,000 light-years from the galactic center. The galaxy is part of the Local "
     "Group, a cluster of galaxies that also includes the Andromeda Galaxy, with "
     "which the Milky Way is expected to collide in roughly four billion years."),
    ("astro-12", "astronomy", "The Big Bang and Cosmic Expansion",
     "The Big Bang theory describes how the universe began about 13.8 billion "
     "years ago as an extremely hot, dense state that has been expanding ever "
     "since. Key evidence includes the cosmic microwave background radiation, the "
     "observed redshift of distant galaxies, and the relative abundance of light "
     "elements like hydrogen and helium. Observations show the expansion is "
     "actually accelerating, which astronomers attribute to a poorly understood "
     "force called dark energy."),
    ("astro-13", "astronomy", "Exoplanets and Habitable Zones",
     "An exoplanet is any planet that orbits a star other than the Sun, and "
     "thousands have been confirmed since the first discoveries in the 1990s, "
     "many detected by the slight dimming they cause as they pass in front of "
     "their host star. The habitable zone is the range of distances from a star "
     "where a planet could maintain liquid water on its surface, a key ingredient "
     "scientists look for when assessing a world's potential to support life."),
    ("astro-14", "astronomy", "Space Telescopes: Hubble and JWST",
     "The Hubble Space Telescope, launched in 1990, orbits above Earth's "
     "atmosphere and has produced some of the most detailed images of distant "
     "galaxies, nebulae, and planets ever captured in visible and ultraviolet "
     "light. The James Webb Space Telescope, launched in 2021, observes primarily "
     "in infrared light, letting it peer through dust clouds and see some of the "
     "earliest galaxies that formed after the Big Bang."),
    ("astro-15", "astronomy", "Human Spaceflight: Apollo to the ISS",
     "The Apollo program landed the first humans on the Moon in 1969, with "
     "astronauts Neil Armstrong and Buzz Aldrin walking on the lunar surface "
     "during Apollo 11. Since the 1990s, the International Space Station has "
     "hosted rotating crews of astronauts conducting continuous research in "
     "microgravity, covering fields from biology to materials science, and "
     "serving as a proving ground for technology needed for future deep-space "
     "missions."),

    # ----------------------------------------------------------------- Nutrition
    ("nutr-01", "nutrition", "Macronutrients: Carbs, Protein, and Fat",
     "Macronutrients are the nutrients the body needs in large amounts for "
     "energy and growth: carbohydrates, protein, and fat. Carbohydrates are the "
     "body's preferred quick-energy source and are broken down into glucose. "
     "Protein supplies amino acids used to build and repair muscle, enzymes, and "
     "other tissues. Fat provides concentrated energy, supports hormone "
     "production, and helps absorb fat-soluble vitamins A, D, E, and K."),
    ("nutr-02", "nutrition", "Vitamins and Their Roles",
     "Vitamins are organic compounds the body needs in small amounts to support "
     "essential chemical reactions. Vitamin C supports immune function and "
     "collagen production, vitamin D helps regulate calcium absorption and bone "
     "health, and B vitamins assist in converting food into usable energy. "
     "Because the body cannot synthesize most vitamins on its own, they must come "
     "from food or, in some cases, sunlight exposure."),
    ("nutr-03", "nutrition", "Essential Minerals in the Diet",
     "Minerals such as calcium, iron, potassium, and zinc are inorganic elements "
     "the body needs for processes like bone formation, oxygen transport, and "
     "immune defense. Calcium and phosphorus make up the structural core of "
     "bones and teeth, while iron is a key component of hemoglobin, the protein "
     "that carries oxygen in red blood cells. Mineral deficiencies, such as low "
     "iron leading to anemia, can cause fatigue and weakened immunity."),
    ("nutr-04", "nutrition", "Dietary Fiber and Gut Health",
     "Dietary fiber is the indigestible part of plant foods that passes largely "
     "intact through the digestive system. Soluble fiber, found in oats and "
     "beans, dissolves in water and can help lower cholesterol, while insoluble "
     "fiber, found in whole grains and vegetable skins, adds bulk that keeps "
     "digestion regular. Fiber also feeds beneficial gut bacteria, which ferment "
     "it into short-chain fatty acids linked to reduced inflammation."),
    ("nutr-05", "nutrition", "Hydration and Daily Water Needs",
     "Water makes up roughly 60 percent of adult body weight and is essential "
     "for regulating temperature, transporting nutrients, and cushioning joints. "
     "Daily needs vary with body size, activity level, and climate, but general "
     "guidelines suggest around two to three liters of total fluid per day from "
     "beverages and food combined. Thirst, urine color, and exercise intensity "
     "are practical signals people use to judge whether they are drinking enough."),
    ("nutr-06", "nutrition", "The Mediterranean Diet",
     "The Mediterranean diet emphasizes vegetables, fruits, whole grains, "
     "legumes, olive oil, and fish, with red meat and sweets eaten only "
     "occasionally. Long-term studies have linked this eating pattern to lower "
     "rates of heart disease and stroke, likely due to its combination of "
     "healthy fats, fiber, and antioxidants. It is often recommended by "
     "dietitians as a sustainable, flavor-forward alternative to more restrictive "
     "diets."),
    ("nutr-07", "nutrition", "Plant-Based Diets",
     "Plant-based diets center on vegetables, fruits, whole grains, legumes, and "
     "nuts, ranging from fully vegan patterns that exclude all animal products to "
     "flexitarian approaches that simply minimize meat. Well-planned plant-based "
     "diets can meet protein needs through combinations like beans and rice, but "
     "may require attention to nutrients that are harder to obtain from plants "
     "alone, such as vitamin B12, iron, and omega-3 fatty acids."),
    ("nutr-08", "nutrition", "The Ketogenic Diet",
     "The ketogenic diet drastically reduces carbohydrate intake, usually below "
     "50 grams a day, while increasing fat to push the body into ketosis, a "
     "metabolic state where it burns fat for fuel and produces ketones. "
     "Originally developed to help manage epilepsy, it has since gained "
     "popularity for weight loss, though it requires careful planning to avoid "
     "nutrient gaps from eliminating many fruits, grains, and legumes."),
    ("nutr-09", "nutrition", "Intermittent Fasting",
     "Intermittent fasting restricts eating to specific time windows rather than "
     "limiting which foods are eaten, with common patterns including a daily "
     "16-hour fast or two reduced-calorie days per week. Proposed benefits "
     "include improved insulin sensitivity and cellular repair processes "
     "triggered during extended fasting periods, though research on long-term "
     "outcomes compared to simple calorie restriction is still developing."),
    ("nutr-10", "nutrition", "Added Sugar and Its Health Effects",
     "Added sugars are sugars and syrups put into foods during processing or "
     "preparation, distinct from sugars naturally present in fruit or milk. "
     "Diets high in added sugar are linked to weight gain, increased risk of "
     "type 2 diabetes, and dental cavities, partly because sugary foods are "
     "often calorie-dense but low in fiber and protein. Health guidelines "
     "generally recommend keeping added sugar below 10 percent of daily "
     "calories."),
    ("nutr-11", "nutrition", "Animal and Plant Protein Sources",
     "Protein sources are often described as complete or incomplete based on "
     "whether they provide all nine essential amino acids in sufficient amounts. "
     "Animal sources like eggs, meat, and dairy are complete proteins, while most "
     "plant sources such as beans or rice are missing or low in at least one "
     "essential amino acid, which is why combining different plant proteins "
     "across the day helps vegetarians meet their needs."),
    ("nutr-12", "nutrition", "Healthy Fats: Omega-3s and Saturated Fat",
     "Not all dietary fats affect the body the same way. Omega-3 fatty acids, "
     "found in fatty fish, walnuts, and flaxseed, support heart and brain health "
     "and have anti-inflammatory effects. Saturated fat, common in red meat and "
     "butter, raises LDL cholesterol when eaten in excess, while unsaturated fats "
     "from olive oil, avocados, and nuts are generally considered better for "
     "cardiovascular health when they replace saturated fat in the diet."),
    ("nutr-13", "nutrition", "Reading Nutrition Facts Labels",
     "Nutrition facts labels list serving size, calories, and amounts of "
     "macronutrients and key micronutrients per serving, along with percent "
     "daily values that show how a serving fits into a typical 2,000-calorie "
     "diet. Checking the serving size first is important, since all the other "
     "numbers on the label scale to it, and a package can contain several "
     "servings even if it looks like a single portion."),
    ("nutr-14", "nutrition", "Meal Timing and Metabolism",
     "Meal timing refers to when food is eaten relative to sleep and activity, "
     "and research suggests it can influence metabolism, blood sugar control, "
     "and appetite regulation independent of total calories consumed. Eating "
     "larger meals earlier in the day and finishing eating several hours before "
     "bedtime has been associated with better blood sugar control in some "
     "studies, though individual responses vary with lifestyle and genetics."),
    ("nutr-15", "nutrition", "Electrolytes and Sports Nutrition",
     "Electrolytes such as sodium, potassium, and magnesium are minerals that "
     "carry an electric charge and are essential for muscle contraction, nerve "
     "signaling, and fluid balance. Prolonged exercise, especially in heat, "
     "causes electrolyte loss through sweat, which is why sports drinks and "
     "electrolyte tablets are formulated to replace sodium and potassium "
     "alongside plain water during long workouts or endurance events."),

    # ------------------------------------------------------------ Programming
    ("prog-01", "programming", "Python: A High-Level Scripting Language",
     "Python is a high-level, dynamically typed programming language known for "
     "readable syntax and a large standard library. It is widely used for data "
     "science, web backends, automation scripts, and machine learning, partly "
     "because libraries like NumPy and pandas make numerical work concise. "
     "Python code is executed by an interpreter, typically CPython, rather than "
     "being compiled directly to native machine code ahead of time."),
    ("prog-02", "programming", "JavaScript and the Web",
     "JavaScript is the primary language that runs inside web browsers, "
     "enabling interactive pages that respond to clicks, form input, and "
     "animations without reloading. It uses an event-driven, single-threaded "
     "model with an event loop that handles asynchronous operations like network "
     "requests. Beyond the browser, the Node.js runtime lets JavaScript run on "
     "servers, which is why it is common across both frontend and backend code."),
    ("prog-03", "programming", "Rust and Memory Safety",
     "Rust is a systems programming language designed to give the performance of "
     "C and C++ while preventing common memory errors like use-after-free bugs "
     "and data races. It achieves this through its ownership and borrowing "
     "system, which the compiler checks at compile time rather than relying on a "
     "garbage collector or runtime checks. This makes Rust popular for "
     "performance-critical software where safety bugs would otherwise be costly."),
    ("prog-04", "programming", "Go and Lightweight Concurrency",
     "Go, created at Google, is a compiled, statically typed language designed "
     "for simplicity and fast builds. Its standout feature is goroutines, "
     "lightweight threads managed by the Go runtime, paired with channels for "
     "passing data safely between them. This makes writing concurrent network "
     "services, such as web servers handling thousands of simultaneous "
     "connections, considerably simpler than with traditional operating-system "
     "threads."),
    ("prog-05", "programming", "Java and the JVM",
     "Java source code compiles to bytecode that runs on the Java Virtual "
     "Machine, which is what makes Java programs portable across operating "
     "systems without recompilation. The JVM includes an automatic garbage "
     "collector that reclaims memory from objects no longer in use, freeing "
     "developers from manual memory management. Java remains widely used in "
     "large enterprise systems, Android app development, and backend services."),
    ("prog-06", "programming", "C and Systems Programming",
     "C is a low-level, compiled language that gives programmers direct control "
     "over memory through pointers and manual allocation with functions like "
     "malloc and free. This control makes C well suited for operating systems, "
     "embedded devices, and performance-critical libraries, but it also means "
     "programmers are responsible for avoiding memory leaks, buffer overflows, "
     "and dangling pointers that the language itself does not prevent."),
    ("prog-07", "programming", "C++ and Object-Oriented Systems Programming",
     "C++ extends C with object-oriented features such as classes, inheritance, "
     "and polymorphism, while retaining low-level memory control and the ability "
     "to compile to efficient native code. It is widely used in game engines, "
     "high-frequency trading systems, and performance-sensitive applications "
     "where both abstraction and raw speed matter. Modern C++ also offers smart "
     "pointers that automate memory cleanup while still avoiding a full garbage "
     "collector."),
    ("prog-08", "programming", "TypeScript and Static Typing for JavaScript",
     "TypeScript is a superset of JavaScript that adds optional static types, "
     "which are checked at compile time and then stripped away when the code is "
     "compiled down to plain JavaScript. Static types let editors catch many "
     "bugs, like passing a string where a number is expected, before the code "
     "ever runs. This has made TypeScript the default choice for many large "
     "frontend and Node.js codebases where maintainability matters."),
    ("prog-09", "programming", "SQL and Relational Databases",
     "SQL, or Structured Query Language, is used to define, query, and modify "
     "data stored in relational databases organized as tables of rows and "
     "columns. A SELECT statement retrieves data, often filtering with WHERE "
     "clauses and combining tables with JOIN operations based on shared keys. "
     "Relational databases enforce structure through schemas and constraints, "
     "which helps keep data consistent as applications grow."),
    ("prog-10", "programming", "Functional Programming Concepts",
     "Functional programming treats computation as the evaluation of "
     "mathematical functions, favoring immutable data and avoiding side "
     "effects wherever possible. Languages like Haskell and Lisp are built "
     "around these ideas, but concepts such as pure functions, map and filter "
     "operations, and higher-order functions have also been adopted by "
     "mainstream languages like Python and JavaScript. Pure functions are easier "
     "to test and reason about because their output depends only on their "
     "input."),
    ("prog-11", "programming", "Version Control with Git",
     "Git is a distributed version control system that tracks changes to source "
     "code over time, letting developers create branches to work on features in "
     "isolation and later merge them back together. Each commit records a "
     "snapshot of the project along with a message describing the change, "
     "forming a history that can be reviewed, reverted, or compared. Being "
     "distributed means every developer has a full copy of the project history, "
     "not just a central server."),
    ("prog-12", "programming", "Static Typing vs Dynamic Typing",
     "In statically typed languages like Java or Rust, variable types are "
     "checked at compile time, catching many type-related errors before the "
     "program runs. In dynamically typed languages like Python or JavaScript, "
     "types are checked at runtime, which speeds up writing small scripts but "
     "can let certain bugs slip through to production. Many teams bridge the gap "
     "with optional type checkers or gradual typing systems added on top of "
     "dynamic languages."),
    ("prog-13", "programming", "Garbage Collection Explained",
     "Garbage collection is an automatic memory management technique that "
     "identifies objects a program no longer uses and frees their memory "
     "without the programmer calling a manual delete. Common strategies include "
     "reference counting, which tracks how many references point to an object, "
     "and tracing collectors that periodically scan for reachable objects from a "
     "set of roots. Languages such as Java, Python, and Go all rely on garbage "
     "collection instead of manual memory management."),
    ("prog-14", "programming", "Compilers vs Interpreters",
     "A compiler translates source code into machine code or an intermediate "
     "form ahead of execution, producing a standalone program that can run "
     "repeatedly without the original compiler present. An interpreter instead "
     "reads and executes source code, or a close representation of it, line by "
     "line at run time. Many modern languages blur the line by compiling to "
     "bytecode that an interpreter or virtual machine then executes, sometimes "
     "with just-in-time compilation for extra speed."),
    ("prog-15", "programming", "Concurrency: Threads vs Async",
     "Thread-based concurrency runs multiple sequences of instructions that the "
     "operating system schedules onto available CPU cores, which works well for "
     "CPU-bound work but adds overhead and complexity around shared state. "
     "Asynchronous, event-loop-based concurrency instead runs many operations on "
     "a single thread, switching between them whenever one is waiting on "
     "something like network I/O, which scales efficiently for workloads "
     "dominated by waiting rather than heavy computation."),

    # ------------------------------------------------------------------ History
    ("hist-01", "history", "Ancient Egypt and the Nile",
     "Ancient Egyptian civilization developed along the Nile River, whose "
     "predictable annual floods deposited fertile silt that supported "
     "large-scale farming. This agricultural surplus allowed the rise of "
     "centralized rule under pharaohs, massive construction projects like the "
     "pyramids at Giza, and a complex system of writing known as hieroglyphics. "
     "Egyptian civilization lasted for roughly three thousand years, making it "
     "one of the longest-lived states in recorded history."),
    ("hist-02", "history", "The Roman Republic and Empire",
     "Rome began as a small city-state and grew into a republic governed by "
     "elected officials and a Senate before eventually transforming into an "
     "empire under Augustus in 27 BCE. At its height, the Roman Empire "
     "controlled territory from Britain to the Middle East, connected by roads, "
     "a shared legal system, and Latin as an administrative language. Its "
     "eventual decline in the west during the fifth century CE is attributed to "
     "a mix of economic strain, military overextension, and repeated invasions."),
    ("hist-03", "history", "The Silk Road Trade Network",
     "The Silk Road was a network of overland and maritime trade routes "
     "connecting China, Central Asia, the Middle East, and Europe for roughly "
     "fifteen hundred years. Merchants carried silk, spices, and precious metals "
     "across these routes, but the network also spread religions, technologies "
     "like papermaking, and diseases such as the plague. No single road existed; "
     "instead goods passed through many hands across a chain of regional "
     "markets."),
    ("hist-04", "history", "The Printing Press and Gutenberg",
     "Johannes Gutenberg introduced a movable-type printing press in Europe "
     "around 1440, dramatically speeding up book production compared to "
     "hand-copying manuscripts. Cheaper, faster printing helped spread literacy, "
     "standardized spelling and language, and accelerated the circulation of "
     "new scientific and religious ideas, including texts that fueled the "
     "Protestant Reformation. Historians often cite the printing press as one of "
     "the most transformative information technologies before the internet."),
    ("hist-05", "history", "The Renaissance",
     "The Renaissance was a period of renewed interest in classical Greek and "
     "Roman art, literature, and philosophy that began in Italian city-states "
     "during the fourteenth century and spread across Europe. Wealthy patrons "
     "funded artists like Leonardo da Vinci and Michelangelo, while scholars "
     "revived humanist ideas that emphasized human potential and observation of "
     "the natural world. The period laid intellectual groundwork for the later "
     "Scientific Revolution."),
    ("hist-06", "history", "The Age of Exploration",
     "Beginning in the fifteenth century, European powers like Portugal and "
     "Spain funded long ocean voyages in search of new trade routes to Asia, "
     "aided by advances in shipbuilding and navigation instruments like the "
     "astrolabe. These voyages led to Christopher Columbus's 1492 arrival in the "
     "Americas and Vasco da Gama's sea route to India, reshaping global trade "
     "but also triggering colonization, exploitation, and the transatlantic "
     "slave trade."),
    ("hist-07", "history", "The Industrial Revolution",
     "The Industrial Revolution began in Britain in the late eighteenth century "
     "as steam power, mechanized textile production, and iron manufacturing "
     "replaced hand tools and manual labor in many industries. It triggered mass "
     "migration from rural farms to growing factory cities, reshaped working "
     "conditions and child labor practices, and set off a sustained rise in "
     "economic output that eventually spread to continental Europe and North "
     "America."),
    ("hist-08", "history", "The French Revolution",
     "The French Revolution began in 1789 amid widespread anger over social "
     "inequality, food shortages, and royal debt, culminating in the storming of "
     "the Bastille prison and the eventual execution of King Louis XVI. The "
     "revolution introduced ideas of popular sovereignty and citizenship that "
     "influenced political movements worldwide, though it also descended into a "
     "violent period known as the Reign of Terror before Napoleon Bonaparte rose "
     "to power."),
    ("hist-09", "history", "Causes of World War I",
     "World War I broke out in 1914 after the assassination of Archduke Franz "
     "Ferdinand of Austria-Hungary, but the underlying causes included a tangle "
     "of military alliances, intense nationalism, and an ongoing arms race among "
     "European powers. Once Austria-Hungary declared war on Serbia, alliance "
     "commitments pulled in Russia, Germany, France, and Britain within weeks, "
     "turning a regional crisis into a continental war fought largely in "
     "entrenched trench lines."),
    ("hist-10", "history", "World War II Overview",
     "World War II began in 1939 when Germany invaded Poland, eventually "
     "drawing in most of the world's nations into two main alliances, the Axis "
     "and the Allies. The war saw unprecedented mobilization of industry, the "
     "Holocaust's systematic genocide of six million Jews and other victims, and "
     "ended in 1945 after Germany's surrender in Europe and the atomic bombings "
     "of Hiroshima and Nagasaki led to Japan's surrender."),
    ("hist-11", "history", "The Cold War",
     "The Cold War was a decades-long geopolitical standoff between the United "
     "States and the Soviet Union following World War II, characterized by "
     "nuclear arms competition, proxy wars in regions like Korea and Vietnam, "
     "and an ideological struggle between capitalism and communism. Rather than "
     "direct military conflict between the two superpowers, the rivalry played "
     "out through espionage, propaganda, and a space race that culminated in the "
     "1969 Moon landing."),
    ("hist-12", "history", "The Fall of the Berlin Wall",
     "The Berlin Wall was built in 1961 to stop East Germans from fleeing to the "
     "West, dividing the city for nearly three decades and becoming a global "
     "symbol of the Cold War's division of Europe. Mounting economic pressure "
     "and political reform across the Eastern Bloc led to its fall in November "
     "1989, when crowds began dismantling sections of the wall, an event that "
     "preceded German reunification less than a year later."),
    ("hist-13", "history", "Ancient Greece and the Birth of Democracy",
     "Ancient Greek city-states, especially Athens, developed early forms of "
     "democratic government in which citizens could vote directly on laws and "
     "policies, though voting rights were limited to free adult men. Greek "
     "philosophers like Socrates, Plato, and Aristotle laid foundations for "
     "Western philosophy, while playwrights and historians produced works still "
     "studied today. Greek city-states frequently fought each other, as in the "
     "Peloponnesian War between Athens and Sparta."),
    ("hist-14", "history", "The Mongol Empire",
     "Founded by Genghis Khan in the early thirteenth century, the Mongol "
     "Empire grew into the largest contiguous land empire in history, stretching "
     "from Korea to Eastern Europe. Mongol rule, despite its brutal conquests, "
     "established the Pax Mongolica, a period of relative stability that "
     "revitalized Silk Road trade and allowed ideas, goods, and technologies to "
     "flow more freely across Eurasia."),
    ("hist-15", "history", "The Neolithic Agricultural Revolution",
     "The Neolithic Revolution, beginning roughly 12,000 years ago, marked "
     "humanity's shift from nomadic hunting and gathering to settled farming and "
     "animal domestication, first in regions like the Fertile Crescent. Farming "
     "allowed food surpluses that supported larger, permanent settlements, "
     "specialized labor, and eventually the first cities, laying the groundwork "
     "for complex civilizations."),

    # -------------------------------------------------------------- Personal finance
    ("fin-01", "finance", "Budgeting with the 50/30/20 Rule",
     "The 50/30/20 rule is a simple budgeting guideline that allocates 50 "
     "percent of after-tax income to needs like housing and groceries, 30 "
     "percent to wants like dining out and entertainment, and 20 percent to "
     "savings and debt repayment. It is popular as a starting framework because "
     "it is easy to remember, though people with high housing costs or heavy "
     "debt often need to adjust the percentages to fit their situation."),
    ("fin-02", "finance", "Building an Emergency Fund",
     "An emergency fund is money set aside in an easily accessible account to "
     "cover unexpected expenses like medical bills, car repairs, or job loss "
     "without resorting to high-interest debt. Financial advisors commonly "
     "recommend saving three to six months of essential living expenses, kept in "
     "a separate savings account rather than invested in the stock market, so it "
     "remains stable and quickly available when needed."),
    ("fin-03", "finance", "How Compound Interest Works",
     "Compound interest is interest calculated on both the original principal "
     "and the interest that has already accumulated, which causes savings or "
     "debt to grow faster over time than simple interest would. A small amount "
     "invested consistently over many years can grow substantially due to "
     "compounding, which is why financial advisors often stress starting to save "
     "for retirement as early as possible, even in small amounts."),
    ("fin-04", "finance", "Understanding Credit Scores",
     "A credit score is a number, typically between 300 and 850, that "
     "summarizes how likely a person is to repay borrowed money based on their "
     "credit history. Factors that influence it include payment history, total "
     "amounts owed, length of credit history, and the mix of credit types used. "
     "A higher score generally qualifies borrowers for lower interest rates on "
     "loans, mortgages, and credit cards."),
    ("fin-05", "finance", "Index Funds and Diversification",
     "An index fund is a pooled investment that holds all or a representative "
     "sample of the securities in a market index, such as a broad stock market "
     "index, aiming to match its performance rather than beat it. Because they "
     "spread money across hundreds or thousands of companies, index funds offer "
     "built-in diversification that reduces the risk of any single company's "
     "poor performance badly damaging a portfolio, usually at a lower cost than "
     "actively managed funds."),
    ("fin-06", "finance", "Retirement Savings Accounts",
     "Tax-advantaged retirement accounts let people set aside money for "
     "retirement while deferring or avoiding taxes on investment growth, "
     "depending on the account type. Contributions to a traditional account "
     "typically reduce taxable income now but are taxed on withdrawal, while "
     "contributions to a Roth-style account are taxed upfront but grow and can "
     "be withdrawn tax-free in retirement. Employer matching contributions, when "
     "offered, effectively add free money to an employee's savings."),
    ("fin-07", "finance", "Debt Payoff: Snowball vs Avalanche",
     "The debt snowball method pays off debts from smallest balance to largest, "
     "regardless of interest rate, to build motivation through quick wins as "
     "each balance disappears. The debt avalanche method instead targets the "
     "debt with the highest interest rate first, which minimizes the total "
     "interest paid over time even though the first payoff may take longer to "
     "achieve. Both methods make minimum payments on all other debts while "
     "focusing extra payments on the targeted one."),
    ("fin-08", "finance", "Inflation and Purchasing Power",
     "Inflation is the rate at which prices for goods and services rise over "
     "time, which erodes the purchasing power of a fixed amount of money. A "
     "modest, steady inflation rate is generally considered normal for a healthy "
     "economy, but high or unpredictable inflation makes planning difficult and "
     "can disproportionately hurt people on fixed incomes or those holding cash "
     "savings that do not earn enough interest to keep pace."),
    ("fin-09", "finance", "The Basics of Insurance",
     "Insurance is a financial arrangement in which a policyholder pays regular "
     "premiums to an insurer in exchange for protection against the cost of "
     "specific risks, such as illness, accidents, or property damage. Insurance "
     "works by pooling risk across many policyholders, since most people will "
     "never file a large claim, allowing the insurer to cover the smaller number "
     "who do. Deductibles and coverage limits determine how much of a loss the "
     "policyholder still bears."),
    ("fin-10", "finance", "Mortgages and Home Loans",
     "A mortgage is a loan used to purchase real estate, where the property "
     "itself serves as collateral that the lender can seize through foreclosure "
     "if payments are not made. Fixed-rate mortgages keep the same interest rate "
     "for the life of the loan, giving predictable payments, while "
     "adjustable-rate mortgages start with a lower rate that can rise or fall "
     "over time based on market conditions."),
    ("fin-11", "finance", "Stock Market Basics",
     "A stock represents partial ownership in a company, and its price "
     "fluctuates based on factors like company earnings, investor sentiment, and "
     "broader economic conditions. Investors can earn returns through price "
     "appreciation if the stock rises in value, or through dividends, which are "
     "periodic payments some companies distribute from their profits. Stock "
     "prices can be volatile in the short term even for fundamentally strong "
     "companies."),
    ("fin-12", "finance", "Bonds and Fixed Income",
     "A bond is essentially a loan an investor makes to a government or "
     "corporation, which agrees to pay periodic interest and return the "
     "principal at a set maturity date. Bonds are generally considered less "
     "risky than stocks because payments are more predictable, but bond prices "
     "fall when interest rates rise, since newly issued bonds then offer more "
     "attractive rates than older ones."),
    ("fin-13", "finance", "Understanding Income Taxes",
     "Income tax is a government levy on earnings, often structured as a "
     "progressive system where higher portions of income are taxed at higher "
     "rates, called tax brackets. Deductions reduce the amount of income subject "
     "to tax, while credits reduce the tax bill directly, which is why a tax "
     "credit is generally more valuable dollar-for-dollar than a deduction of the "
     "same size."),
    ("fin-14", "finance", "Dollar-Cost Averaging",
     "Dollar-cost averaging is an investment strategy where a fixed amount of "
     "money is invested at regular intervals, regardless of whether prices are "
     "high or low at the time. This approach naturally buys more shares when "
     "prices are low and fewer when prices are high, smoothing out the impact of "
     "short-term market volatility compared to investing a lump sum all at once."),
    ("fin-15", "finance", "Common Behavioral Finance Biases",
     "Behavioral finance studies how psychological biases cause investors to "
     "make decisions that deviate from purely rational models. Loss aversion "
     "describes the tendency to feel losses more intensely than equivalent "
     "gains, which can cause investors to hold losing positions too long. "
     "Herd behavior, following what others are doing rather than independent "
     "analysis, can amplify market bubbles and crashes."),

    # ------------------------------------------------------------- Machine learning
    ("ml-01", "machine_learning", "Supervised Learning Overview",
     "Supervised learning trains a model on labeled examples, pairs of inputs "
     "and known correct outputs, so the model can learn a mapping it can later "
     "apply to new, unseen inputs. Classification tasks predict discrete "
     "categories, such as spam versus not spam, while regression tasks predict "
     "continuous values, such as a house price. The quality and quantity of "
     "labeled training data strongly influence how well a supervised model "
     "performs."),
    ("ml-02", "machine_learning", "Unsupervised Learning and Clustering",
     "Unsupervised learning works with data that has no labeled outcomes, "
     "instead looking for structure or patterns on its own. Clustering "
     "algorithms, such as k-means, group similar data points together based on "
     "distance or similarity measures, which is useful for tasks like customer "
     "segmentation. Dimensionality reduction techniques like PCA are also "
     "unsupervised, compressing data into fewer variables while preserving as "
     "much information as possible."),
    ("ml-03", "machine_learning", "Neural Network Basics",
     "A neural network is composed of layers of interconnected nodes, or "
     "neurons, each applying a weighted sum of its inputs followed by a "
     "nonlinear activation function. Stacking multiple layers lets the network "
     "learn increasingly complex patterns, which is the basis of deep learning. "
     "Training adjusts the weights using an algorithm called backpropagation, "
     "which calculates how much each weight contributed to the overall "
     "prediction error."),
    ("ml-04", "machine_learning", "Overfitting and Regularization",
     "Overfitting occurs when a model learns patterns specific to the training "
     "data, including its noise, so well that it performs poorly on new data it "
     "has not seen before. Regularization techniques, such as adding a penalty "
     "for large weights or randomly dropping neurons during training with "
     "dropout, discourage the model from relying too heavily on any single "
     "feature. Collecting more training data and simplifying the model are other "
     "common ways to reduce overfitting."),
    ("ml-05", "machine_learning", "Gradient Descent and Optimization",
     "Gradient descent is an optimization algorithm that iteratively adjusts a "
     "model's parameters to minimize a loss function, moving each parameter in "
     "the direction that most reduces error. The learning rate controls how "
     "large each adjustment step is; too large a rate can overshoot the minimum, "
     "while too small a rate makes training slow. Variants like stochastic "
     "gradient descent update parameters using small random batches of data "
     "rather than the entire dataset at once, speeding up training."),
    ("ml-06", "machine_learning", "Decision Trees and Random Forests",
     "A decision tree splits data repeatedly based on feature values, forming a "
     "tree of yes-or-no questions that leads to a prediction at each leaf. "
     "Individual decision trees can overfit easily, so random forests combine "
     "many trees, each trained on a random subset of data and features, and "
     "average their predictions to produce a more robust and accurate model."),
    ("ml-07", "machine_learning", "Convolutional Neural Networks",
     "Convolutional neural networks, or CNNs, are designed for grid-like data "
     "such as images, using small filters that slide across the input to detect "
     "local patterns like edges and textures. Early layers typically learn "
     "simple features, while deeper layers combine them into more complex "
     "shapes and objects. CNNs power most modern image classification, object "
     "detection, and facial recognition systems."),
    ("ml-08", "machine_learning", "Recurrent Neural Networks and Sequences",
     "Recurrent neural networks process sequential data, such as text or time "
     "series, by maintaining a hidden state that carries information from "
     "previous time steps forward as new inputs arrive. Variants like LSTM and "
     "GRU cells were designed to better preserve information over long "
     "sequences, addressing the vanishing gradient problem that made plain RNNs "
     "struggle to learn long-range dependencies."),
    ("ml-09", "machine_learning", "Transformers and Attention",
     "The transformer architecture relies on an attention mechanism that lets "
     "the model weigh how relevant every other token in a sequence is when "
     "processing a given token, rather than passing information step by step as "
     "recurrent networks do. This allows transformers to be trained in parallel "
     "across an entire sequence and to capture long-range relationships "
     "effectively, which is why they underpin most modern large language models "
     "and many embedding models used for search."),
    ("ml-10", "machine_learning", "Train, Validation, and Test Splits",
     "Machine learning datasets are typically divided into a training set used "
     "to fit the model, a validation set used to tune hyperparameters and "
     "compare models during development, and a held-out test set used only once "
     "to estimate how the final model will perform on new data. Keeping the test "
     "set untouched during development is essential to get an honest estimate of "
     "real-world performance."),
    ("ml-11", "machine_learning", "Feature Engineering",
     "Feature engineering is the process of transforming raw data into inputs "
     "that better expose the underlying patterns a model needs to learn, such as "
     "extracting the day of the week from a timestamp or combining two columns "
     "into a meaningful ratio. Even with modern deep learning, which can learn "
     "some features automatically, thoughtful feature engineering often still "
     "improves performance, especially for structured, tabular data."),
    ("ml-12", "machine_learning", "Cross-Validation",
     "Cross-validation evaluates how well a model generalizes by repeatedly "
     "splitting the training data into different training and validation "
     "subsets and averaging the resulting performance scores. K-fold "
     "cross-validation, a common approach, divides the data into k equally "
     "sized parts, trains on k-minus-one of them, and validates on the "
     "remaining part, rotating through all k combinations to get a more reliable "
     "performance estimate than a single train-validation split."),
    ("ml-13", "machine_learning", "The Bias-Variance Tradeoff",
     "The bias-variance tradeoff describes the balance between a model that is "
     "too simple, which underfits the data with high bias, and a model that is "
     "too complex, which overfits the data with high variance to noise in the "
     "training set. The goal is to find a model complexity that minimizes total "
     "error on new data by balancing both sources of error rather than "
     "minimizing either one alone."),
    ("ml-14", "machine_learning", "Reinforcement Learning Basics",
     "Reinforcement learning trains an agent to make decisions by interacting "
     "with an environment and receiving rewards or penalties based on the "
     "outcomes of its actions, gradually learning a policy that maximizes "
     "cumulative reward over time. Unlike supervised learning, there is no fixed "
     "set of correct labeled answers; instead, the agent must explore different "
     "actions to discover which strategies work best, a tension often called the "
     "exploration-exploitation tradeoff."),
    ("ml-15", "machine_learning", "Embeddings and Vector Representations",
     "An embedding is a way of representing data, such as words, sentences, or "
     "images, as dense numerical vectors in a continuous space where similar "
     "items end up positioned close together. Embedding models are typically "
     "trained so that semantic similarity in meaning corresponds to geometric "
     "closeness, often measured with cosine similarity, which is the foundation "
     "that dense vector search and recommendation systems rely on."),

    # -------------------------------------------------------------------- Climate
    ("clim-01", "climate", "The Greenhouse Effect",
     "The greenhouse effect occurs when gases in the atmosphere, such as carbon "
     "dioxide and methane, trap heat radiated from the Earth's surface instead "
     "of letting it escape directly into space. This natural effect keeps the "
     "planet warm enough to support life, but human activities like burning "
     "fossil fuels have sharply increased greenhouse gas concentrations, "
     "intensifying the effect and driving long-term global warming."),
    ("clim-02", "climate", "The Carbon Cycle",
     "The carbon cycle describes how carbon moves between the atmosphere, "
     "oceans, soil, and living organisms through processes like photosynthesis, "
     "respiration, and the decomposition of organic matter. Burning fossil "
     "fuels releases carbon that had been locked underground for millions of "
     "years, adding it back into the atmosphere far faster than natural "
     "processes can reabsorb it, which disrupts the cycle's long-standing "
     "balance."),
    ("clim-03", "climate", "Solar Power as Renewable Energy",
     "Solar power converts sunlight into electricity using photovoltaic cells "
     "that generate an electric current when struck by photons, or through "
     "concentrated solar systems that use mirrors to heat a fluid that drives a "
     "turbine. Solar energy produces no direct emissions during operation and "
     "costs have fallen sharply over the past decade, though output varies with "
     "weather, time of day, and season, which is why energy storage is often "
     "paired with solar installations."),
    ("clim-04", "climate", "Wind Power as Renewable Energy",
     "Wind power uses turbines with large rotating blades to convert the "
     "kinetic energy of moving air into electricity through a generator. Wind "
     "farms can be built on land or offshore, where ocean winds tend to be "
     "stronger and more consistent, and like solar power, wind generates "
     "electricity without direct greenhouse gas emissions. Its main limitation "
     "is intermittency, since output depends on wind speed and can vary "
     "significantly hour to hour."),
    ("clim-05", "climate", "Ocean Acidification",
     "Ocean acidification occurs as the ocean absorbs excess atmospheric carbon "
     "dioxide, which reacts with seawater to form carbonic acid and lowers the "
     "ocean's pH over time. More acidic water makes it harder for shellfish, "
     "corals, and some plankton to build their calcium carbonate shells and "
     "skeletons, threatening marine food webs that many ocean species and "
     "coastal communities depend on."),
    ("clim-06", "climate", "Deforestation and Its Impacts",
     "Deforestation is the large-scale clearing of forests, often to make room "
     "for agriculture, grazing land, or logging, and it removes trees that would "
     "otherwise absorb carbon dioxide from the atmosphere. Beyond releasing "
     "stored carbon when cleared or burned, deforestation destroys habitats for "
     "countless species and can disrupt regional rainfall patterns, since large "
     "forests like the Amazon influence the water cycle well beyond their own "
     "boundaries."),
    ("clim-07", "climate", "Biodiversity Loss",
     "Biodiversity loss refers to the ongoing decline in the variety of species "
     "and genetic diversity within ecosystems, driven largely by habitat "
     "destruction, climate change, pollution, and overexploitation of wildlife. "
     "Scientists warn that current extinction rates are far above natural "
     "background levels, and losing species can weaken ecosystem resilience, "
     "since diverse ecosystems are generally better able to withstand "
     "disturbances like disease or drought."),
    ("clim-08", "climate", "Sea Level Rise",
     "Sea levels are rising due to two main factors linked to global warming: "
     "thermal expansion of ocean water as it warms, and the melting of land ice "
     "from glaciers and ice sheets in places like Greenland and Antarctica. "
     "Rising seas increase the risk of coastal flooding, erode shorelines, and "
     "can contaminate freshwater supplies with saltwater, threatening "
     "low-lying island nations and coastal cities around the world."),
    ("clim-09", "climate", "Extreme Weather and a Warming Climate",
     "A warmer atmosphere holds more moisture and carries more energy, which "
     "scientists link to an increase in the frequency and intensity of extreme "
     "weather events such as heavy rainfall, prolonged droughts, and powerful "
     "hurricanes. While no single storm or heatwave can be blamed entirely on "
     "climate change, statistical analysis increasingly shows that global "
     "warming makes many extreme events more likely or more severe than they "
     "would otherwise be."),
    ("clim-10", "climate", "Carbon Footprint and Emissions",
     "A carbon footprint is the total amount of greenhouse gases, usually "
     "expressed as carbon dioxide equivalent, generated by an individual, "
     "organization, product, or event. Major contributors to a typical "
     "individual's footprint include transportation, home energy use, and diet, "
     "with air travel and red meat consumption often cited as particularly "
     "carbon-intensive choices that people can target to reduce their impact."),
    ("clim-11", "climate", "Recycling and the Circular Economy",
     "Recycling reprocesses used materials like paper, glass, and certain "
     "plastics into new products, reducing the need to extract and process "
     "virgin raw materials. The broader idea of a circular economy goes further, "
     "aiming to design products and systems so materials are reused, repaired, "
     "or recycled indefinitely rather than following the traditional "
     "take-make-dispose model, minimizing waste sent to landfills."),
    ("clim-12", "climate", "Air Pollution and Human Health",
     "Air pollution from vehicle exhaust, industrial emissions, and burning "
     "fossil fuels releases particulate matter and gases like nitrogen dioxide "
     "that can penetrate deep into the lungs and bloodstream. Long-term exposure "
     "is linked to respiratory diseases, cardiovascular problems, and reduced "
     "life expectancy, with children, the elderly, and people with existing "
     "health conditions typically facing the greatest risk."),
    ("clim-13", "climate", "Climate Policy and the Paris Agreement",
     "The Paris Agreement, adopted in 2015, is an international treaty in which "
     "nearly every country committed to limiting global temperature rise, aiming "
     "to keep warming well below 2 degrees Celsius above pre-industrial levels. "
     "Each participating country sets its own voluntary emissions targets, "
     "called nationally determined contributions, and reports progress "
     "periodically, though the agreement itself has no mechanism to directly "
     "penalize countries that fall short."),
    ("clim-14", "climate", "Permafrost Thaw",
     "Permafrost is ground that has remained frozen for at least two "
     "consecutive years, found across large areas of the Arctic, and it stores "
     "enormous amounts of ancient organic carbon. As rising temperatures cause "
     "permafrost to thaw, buried organic material decomposes and releases carbon "
     "dioxide and methane, a potent greenhouse gas, creating a feedback loop that "
     "could further accelerate warming."),
    ("clim-15", "climate", "Coral Reef Bleaching",
     "Coral reefs host colorful algae that live inside coral tissue and provide "
     "most of the coral's energy and color through photosynthesis. When ocean "
     "water becomes too warm, corals expel these algae in a stress response "
     "called bleaching, leaving the coral white and vulnerable to starvation and "
     "disease. Repeated or prolonged bleaching events, increasingly common as "
     "oceans warm, can kill entire reef systems that took centuries to form."),

    # -------------------------------------------------------------------- Cooking
    ("cook-01", "cooking", "Basic Knife Skills",
     "Good knife skills start with a proper grip, holding the blade with a "
     "pinch grip near the handle while curling the guiding hand's fingers under "
     "to protect them, a technique often called the claw grip. Common cuts "
     "include the julienne, thin matchstick strips, and the dice, uniform cubes "
     "used for even cooking. A sharp knife is actually safer than a dull one, "
     "since it requires less force and is less likely to slip."),
    ("cook-02", "cooking", "Sautéing Technique",
     "Sautéing cooks food quickly in a small amount of fat over relatively high "
     "heat, usually in a wide, shallow pan that allows moisture to evaporate "
     "rather than steam the food. Keeping ingredients moving, either by tossing "
     "the pan or stirring, promotes even browning and prevents sticking. The "
     "technique works best with tender, small pieces of vegetables, meat, or "
     "seafood that cook through in just a few minutes."),
    ("cook-03", "cooking", "Braising Technique",
     "Braising combines dry and wet cooking methods, first searing food at high "
     "heat to develop flavor through browning, then simmering it slowly in a "
     "small amount of liquid in a covered pot. The long, gentle cooking breaks "
     "down tough connective tissue in cuts like beef chuck or pork shoulder, "
     "turning them tender while the surrounding liquid reduces into a flavorful "
     "sauce."),
    ("cook-04", "cooking", "Baking Bread with Yeast",
     "Yeast bread relies on live yeast consuming sugars in the dough and "
     "releasing carbon dioxide gas, which gets trapped in strands of gluten "
     "formed as the dough is kneaded, causing the bread to rise. Allowing the "
     "dough to rest, or proof, gives the yeast time to work and develops flavor "
     "through fermentation. Baking at high heat then sets the structure and "
     "creates a browned crust through the Maillard reaction."),
    ("cook-05", "cooking", "Emulsification: Mayonnaise and Vinaigrette",
     "Emulsification combines two liquids that normally do not mix, such as oil "
     "and water, into a stable blend by dispersing tiny droplets of one liquid "
     "throughout the other. Mayonnaise achieves this with egg yolk, whose "
     "lecithin acts as an emulsifier, while a vinaigrette relies on vigorous "
     "whisking or mustard to temporarily suspend oil droplets in vinegar before "
     "they separate again."),
    ("cook-06", "cooking", "The Maillard Reaction and Searing",
     "The Maillard reaction is a chemical reaction between amino acids and "
     "sugars that occurs when food is heated to around 300 degrees Fahrenheit "
     "or higher, producing the complex browned flavors and aromas associated "
     "with seared meat, toasted bread, and roasted coffee. Achieving a good sear "
     "requires a dry surface and high, consistent heat, since excess surface "
     "moisture cools the pan and causes steaming instead of browning."),
    ("cook-07", "cooking", "Fermentation Basics",
     "Fermentation uses microorganisms like bacteria, yeast, or molds to "
     "transform food, producing byproducts such as acids, alcohol, or gases that "
     "change flavor, texture, and preservation. Lactic acid fermentation, used "
     "in sauerkraut and yogurt, relies on bacteria converting sugars into lactic "
     "acid, which lowers pH enough to inhibit spoilage organisms and gives "
     "fermented foods their characteristic tang."),
    ("cook-08", "cooking", "Making Stock and Broth",
     "Stock is made by simmering bones, and often vegetables and aromatics, in "
     "water for an extended time to extract collagen, flavor compounds, and "
     "minerals, while broth is typically simmered with meat for a shorter time "
     "and has a lighter body. Simmering gently rather than boiling keeps the "
     "liquid clear by preventing fat and proteins from emulsifying into it, and "
     "skimming off foam further improves clarity."),
    ("cook-09", "cooking", "Seasoning with Salt",
     "Salt enhances flavor by suppressing bitterness and amplifying other "
     "tastes, which is why seasoning food at multiple stages of cooking, rather "
     "than only at the end, tends to produce more balanced flavor throughout. "
     "Salt also draws moisture out of food through osmosis, which is useful for "
     "techniques like dry-brining meat or drawing excess water out of salted "
     "vegetables before cooking."),
    ("cook-10", "cooking", "Making Fresh Pasta",
     "Fresh pasta dough is typically made from flour, eggs, and sometimes a "
     "touch of oil or water, kneaded until smooth and elastic as gluten "
     "develops. Resting the dough, covered, for at least thirty minutes relaxes "
     "the gluten, making it easier to roll thin without tearing. Fresh pasta "
     "cooks much faster than dried pasta, often in just one to three minutes in "
     "boiling water."),
    ("cook-11", "cooking", "Grilling vs Roasting",
     "Grilling cooks food directly over high, often open-flame heat, producing "
     "characteristic char marks and smoky flavor in a relatively short time, "
     "which suits thinner cuts of meat and vegetables. Roasting instead surrounds "
     "food with hot, circulating air in an oven, cooking it more evenly and "
     "gently over a longer period, making it better suited to larger cuts like a "
     "whole chicken or a roast that need time for heat to penetrate to the "
     "center."),
    ("cook-12", "cooking", "Sous Vide Cooking",
     "Sous vide cooking seals food in an airtight bag and immerses it in a "
     "water bath held at a precise, constant temperature, often matching the "
     "exact final temperature desired for the food. Because the water "
     "temperature never exceeds the target, it is nearly impossible to overcook "
     "the interior of the food, though a quick sear afterward is usually needed "
     "to develop browning and surface texture that sous vide alone cannot "
     "produce."),
    ("cook-13", "cooking", "Pastry and Gluten Development",
     "Pastry recipes often aim to limit gluten development to produce a tender, "
     "flaky texture rather than a chewy one, which is why pastry dough is mixed "
     "minimally and kept cold so the fat stays in solid pieces. Cold butter cut "
     "into flour creates distinct layers that, when baked, release steam and "
     "separate into the flaky layers characteristic of good pie crust or "
     "croissants."),
    ("cook-14", "cooking", "Building Spice Blends",
     "Spice blends combine multiple dried spices to create layered, balanced "
     "flavor profiles, often grouped by the roles they play, such as warmth from "
     "cinnamon or cumin, heat from chili, and brightness from citrus-forward "
     "spices like sumac. Toasting whole spices briefly before grinding them "
     "releases aromatic oils and deepens their flavor compared to using "
     "pre-ground spices that have already lost much of their potency."),
    ("cook-15", "cooking", "Food Safety and Storage Temperatures",
     "Perishable food should generally be kept out of the temperature danger "
     "zone between 40 and 140 degrees Fahrenheit, the range where bacteria "
     "multiply fastest, which is why refrigerators are kept at or below 40 "
     "degrees. Cooking meat to a safe internal temperature, confirmed with a "
     "food thermometer rather than guesswork, is one of the most reliable ways "
     "to kill harmful bacteria before eating."),
]


def main() -> None:
    out_path = (
        Path(__file__).resolve().parent.parent
        / "src"
        / "semantic_search_engine"
        / "data"
        / "corpus.json"
    )
    payload = [
        {"id": doc_id, "topic": topic, "title": title, "text": text}
        for doc_id, topic, title, text in DOCUMENTS
    ]
    out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Wrote {len(payload)} documents to {out_path}")


if __name__ == "__main__":
    main()
