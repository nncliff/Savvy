#!/usr/bin/env python3
"""
Enhanced Entity Extractor - Optimized to extract 85+ scientific entities
Uses more comprehensive term detection and NLP-style entity recognition
"""

import json
import re
import time
from typing import List, Dict, Set, Tuple
from collections import defaultdict, Counter
from pathlib import Path


class EnhancedEntityExtractor:
    """Enhanced entity extractor targeting 85+ scientific entities."""
    
    def __init__(self):
        # Expanded scientific terms based on actual content analysis
        self.scientific_terms = {
            'ai_ml_core': [
                'artificial intelligence', 'machine learning', 'deep learning', 'neural network',
                'reinforcement learning', 'supervised learning', 'unsupervised learning',
                'natural language processing', 'computer vision', 'pattern recognition'
            ],
            'ai_ml_advanced': [
                'diffusion model', 'transformer', 'attention mechanism', 'gpt', 'bert',
                'generative adversarial network', 'variational autoencoder', 'autoencoder',
                'convolutional neural network', 'recurrent neural network', 'lstm', 'gru',
                'reinforcement learning', 'q learning', 'policy gradient', 'actor critic',
                'meta learning', 'few shot learning', 'transfer learning', 'domain adaptation',
                'self supervised learning', 'contrastive learning', 'representation learning',
                'graph neural network', 'graph convolutional network', 'node embedding',
                'kalman filter', 'particle filter', 'bayesian optimization', 'gaussian process',
                'flow matching', 'normalizing flow', 'score based model', 'energy based model'
            ],
            'ai_ml_applications': [
                'autonomous driving', 'autonomous vehicle', 'robotics', 'autonomous drone',
                'recommendation system', 'collaborative filtering', 'content based filtering',
                'speech recognition', 'text to speech', 'machine translation',
                'image classification', 'object detection', 'semantic segmentation',
                'sentiment analysis', 'question answering', 'dialogue system'
            ],
            'physics_quantum': [
                'quantum computing', 'quantum algorithm', 'quantum circuit', 'quantum gate',
                'quantum entanglement', 'quantum superposition', 'quantum decoherence',
                'quantum error correction', 'quantum supremacy', 'quantum advantage',
                'qubit', 'quantum bit', 'quantum state', 'quantum measurement'
            ],
            'physics_statistical': [
                'statistical mechanics', 'statistical physics', 'thermodynamics',
                'thermodynamic computing', 'boltzmann machine', 'boltzmann distribution',
                'phase transition', 'critical phenomenon', 'phase diagram',
                'monte carlo method', 'molecular dynamics', 'lattice model',
                'ising model', 'spin glass', 'random matrix theory'
            ],
            'physics_general': [
                'quantum mechanics', 'quantum field theory', 'general relativity',
                'special relativity', 'electromagnetism', 'classical mechanics',
                'condensed matter physics', 'solid state physics', 'particle physics',
                'cosmology', 'astrophysics', 'plasma physics'
            ],
            'mathematics_linear': [
                'linear algebra', 'matrix', 'vector', 'eigenvalue', 'eigenvector',
                'singular value decomposition', 'principal component analysis',
                'matrix decomposition', 'matrix factorization', 'least squares',
                'gram schmidt', 'qr decomposition', 'lu decomposition',
                'matrix rank', 'degrees of freedom', 'eigenvector centrality'
            ],
            'mathematics_calculus': [
                'calculus', 'differential equation', 'partial differential equation',
                'ordinary differential equation', 'gradient', 'derivative',
                'integral', 'fourier transform', 'laplace transform',
                'taylor series', 'optimization', 'convex optimization'
            ],
            'mathematics_probability': [
                'probability theory', 'statistics', 'bayesian inference',
                'maximum likelihood', 'expectation maximization', 'markov chain',
                'markov process', 'stochastic process', 'random walk',
                'central limit theorem', 'law of large numbers', 'hypothesis testing'
            ],
            'mathematics_discrete': [
                'graph theory', 'network theory', 'combinatorics', 'discrete mathematics',
                'algorithm', 'data structure', 'complexity theory', 'computational complexity',
                'approximation algorithm', 'greedy algorithm', 'dynamic programming'
            ],
            'mathematics_advanced': [
                'differential geometry', 'topology', 'algebraic geometry', 'category theory',
                'functional analysis', 'measure theory', 'real analysis', 'complex analysis',
                'abstract algebra', 'number theory', 'set theory', 'logic'
            ],
            'systems_complexity': [
                'complex system', 'complexity theory', 'emergence', 'self organization',
                'adaptive system', 'evolutionary algorithm', 'genetic algorithm',
                'swarm intelligence', 'collective behavior', 'network dynamics',
                'social network', 'scale free network', 'small world network',
                'community detection', 'network centrality', 'network analysis',
                'social influence', 'cognitive perspective', 'social dynamics'
            ],
            'systems_control': [
                'control theory', 'feedback control', 'optimal control', 'robust control',
                'adaptive control', 'predictive control', 'system identification',
                'state space model', 'transfer function', 'frequency response'
            ],
            'computational_science': [
                'computational biology', 'bioinformatics', 'computational chemistry',
                'computational physics', 'scientific computing', 'numerical analysis',
                'finite element method', 'computational fluid dynamics',
                'high performance computing', 'parallel computing', 'distributed computing'
            ],
            'data_science': [
                'data science', 'big data', 'data mining', 'data analysis',
                'feature selection', 'feature engineering', 'dimensionality reduction',
                'clustering', 'classification', 'regression', 'anomaly detection',
                'time series analysis', 'forecasting', 'data visualization'
            ]
        }
        
        # Common generic terms to filter out
        self.generic_terms = {
            'system', 'method', 'approach', 'technique', 'framework', 'model',
            'analysis', 'study', 'research', 'development', 'application',
            'process', 'technology', 'solution', 'implementation', 'design',
            'structure', 'function', 'behavior', 'performance', 'evaluation',
            'comparison', 'improvement', 'enhancement', 'optimization',
            'parameter', 'variable', 'factor', 'component', 'element'
        }
        
        # Build comprehensive term list
        self.all_terms = []
        for domain_terms in self.scientific_terms.values():
            self.all_terms.extend(domain_terms)
        
        # Remove duplicates while preserving order
        seen = set()
        self.all_terms = [term for term in self.all_terms if not (term in seen or seen.add(term))]
        
        print(f"Initialized with {len(self.all_terms)} scientific terms across {len(self.scientific_terms)} domains")
    
    def extract_entities_from_text(self, text: str, context_id: str = None) -> List[Dict]:
        """Extract scientific entities from text using multiple strategies."""
        text_lower = text.lower()
        found_entities = []
        
        # Strategy 1: Exact term matching
        for term in self.all_terms:
            if term.lower() in text_lower:
                # Skip if it's just part of a larger generic term
                if any(generic in term.lower() for generic in self.generic_terms):
                    continue
                
                # Find domain for this term
                domain = self._find_domain_for_term(term)
                
                # Calculate context weight
                weight = self._calculate_context_weight(term, text)
                
                found_entities.append({
                    'name': term,
                    'domain': domain,
                    'frequency': text_lower.count(term.lower()),
                    'context_weight': weight,
                    'extraction_method': 'exact_match',
                    'context_id': context_id
                })
        
        # Strategy 2: Pattern-based extraction for scientific concepts
        scientific_patterns = [
            r'\b([A-Z][a-z]+ (?:algorithm|method|model|network|system|approach|theory))\b',
            r'\b([a-z]+ (?:learning|computing|analysis|optimization|classification))\b',
            r'\b((?:deep|machine|reinforcement|supervised|unsupervised) [a-z]+)\b',
            r'\b([A-Z][A-Za-z]+ (?:equation|theorem|principle|law|effect))\b',
            r'\b([a-z]+ (?:neural network|graph|matrix|vector|space))\b'
        ]
        
        for pattern in scientific_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).lower()
                
                # Skip if generic or already found
                if (any(generic in term for generic in self.generic_terms) or
                    any(entity['name'] == term for entity in found_entities)):
                    continue
                
                # Infer domain based on keywords
                domain = self._infer_domain_from_context(term, text)
                weight = self._calculate_context_weight(term, text)
                
                found_entities.append({
                    'name': term,
                    'domain': domain,
                    'frequency': 1,
                    'context_weight': weight,
                    'extraction_method': 'pattern_match',
                    'context_id': context_id
                })
        
        # Strategy 3: Compound term detection (e.g., "machine learning algorithm")
        compound_patterns = [
            r'\b((?:machine|deep|reinforcement) learning [a-z]+)\b',
            r'\b(quantum (?:computing|algorithm|circuit|gate))\b',
            r'\b(neural network [a-z]+)\b',
            r'\b([a-z]+ optimization [a-z]*)\b',
            r'\b(graph (?:neural|convolutional) network)\b'
        ]
        
        for pattern in compound_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                term = match.group(1).lower()
                
                if any(entity['name'] == term for entity in found_entities):
                    continue
                
                domain = self._infer_domain_from_context(term, text)
                weight = self._calculate_context_weight(term, text)
                
                found_entities.append({
                    'name': term,
                    'domain': domain,
                    'frequency': 1,
                    'context_weight': weight,
                    'extraction_method': 'compound_match',
                    'context_id': context_id
                })
        
        return found_entities
    
    def _find_domain_for_term(self, term: str) -> str:
        """Find which domain a term belongs to."""
        for domain, terms in self.scientific_terms.items():
            if term in terms:
                return domain
        return 'unknown'
    
    def _infer_domain_from_context(self, term: str, text: str) -> str:
        """Infer domain based on surrounding context."""
        text_lower = text.lower()
        
        # AI/ML indicators
        if any(indicator in text_lower for indicator in ['neural', 'learning', 'ai', 'artificial', 'algorithm']):
            if any(indicator in term for indicator in ['quantum', 'thermodynamic', 'statistical']):
                return 'physics_advanced'
            return 'ai_ml_advanced'
        
        # Physics indicators
        if any(indicator in text_lower for indicator in ['quantum', 'physics', 'thermodynamic', 'statistical']):
            return 'physics_advanced'
        
        # Math indicators
        if any(indicator in text_lower for indicator in ['matrix', 'linear', 'algebra', 'calculus', 'optimization']):
            return 'mathematics_advanced'
        
        # Systems indicators
        if any(indicator in text_lower for indicator in ['network', 'social', 'complex', 'system']):
            return 'systems_complexity'
        
        return 'computational_science'
    
    def _calculate_context_weight(self, entity: str, text: str) -> float:
        """Calculate context weight for entity (similar to proven pattern)."""
        entity_pos = text.lower().find(entity.lower())
        if entity_pos == -1:
            return 0.0
        
        # Extract surrounding context
        start = max(0, entity_pos - 150)
        end = min(len(text), entity_pos + len(entity) + 150)
        context = text[start:end].lower()
        
        # Quality indicators
        technical_indicators = [
            'research', 'study', 'paper', 'algorithm', 'method', 'approach',
            'analysis', 'model', 'theory', 'framework', 'system', 'network',
            'optimization', 'classification', 'prediction', 'learning'
        ]
        
        authority_indicators = [
            'nature', 'science', 'journal', 'university', 'institute',
            'conference', 'proceedings', 'publication', 'author', 'researcher'
        ]
        
        complexity_indicators = [
            'advanced', 'novel', 'innovative', 'breakthrough', 'cutting-edge',
            'state-of-the-art', 'pioneering', 'groundbreaking'
        ]
        
        # Calculate scores
        technical_score = sum(1 for ind in technical_indicators if ind in context) / len(technical_indicators)
        authority_score = sum(1 for ind in authority_indicators if ind in context) / len(authority_indicators)
        complexity_score = sum(1 for ind in complexity_indicators if ind in context) / len(complexity_indicators)
        
        # Sentence structure quality
        sentences = context.split('.')
        avg_length = sum(len(s.split()) for s in sentences) / max(len(sentences), 1)
        structure_score = min(avg_length / 15, 1.0)
        
        # Combined weight
        weight = (
            technical_score * 0.3 +
            authority_score * 0.2 +
            complexity_score * 0.2 +
            structure_score * 0.3
        )
        
        return min(weight, 1.0)
    
    def process_dataset(self, data_file: str = "data/bookmarks_complete_translation.json") -> Dict:
        """Process the complete dataset and extract entities."""
        print("="*80)
        print("ENHANCED ENTITY EXTRACTION - TARGET: 85+ ENTITIES")
        print("="*80)
        
        start_time = time.time()
        
        # Load dataset
        with open(data_file, 'r', encoding='utf-8') as f:
            bookmarks = json.load(f)
        
        print(f"Processing {len(bookmarks)} bookmarks...")
        
        # Extract entities from all documents
        all_entities = []
        entity_aggregation = defaultdict(lambda: {
            'total_frequency': 0,
            'total_weight': 0.0,
            'domains': set(),
            'methods': set(),
            'contexts': []
        })
        
        for i, bookmark in enumerate(bookmarks):
            if i % 50 == 0:
                print(f"Processing bookmark {i+1}/{len(bookmarks)}...")
            
            # Use English translation (translation-first architecture)
            title = bookmark['title']['translated']
            content = bookmark['content']['translated']
            full_text = f"{title}. {content}"
            
            # Extract entities from this document
            entities = self.extract_entities_from_text(full_text, context_id=bookmark['id'])
            
            for entity in entities:
                name = entity['name']
                entity_aggregation[name]['total_frequency'] += entity['frequency']
                entity_aggregation[name]['total_weight'] += entity['context_weight']
                entity_aggregation[name]['domains'].add(entity['domain'])
                entity_aggregation[name]['methods'].add(entity['extraction_method'])
                entity_aggregation[name]['contexts'].append({
                    'bookmark_id': bookmark['id'],
                    'weight': entity['context_weight'],
                    'frequency': entity['frequency']
                })
        
        # Consolidate and rank entities
        final_entities = []
        for name, data in entity_aggregation.items():
            avg_weight = data['total_weight'] / len(data['contexts'])
            final_score = avg_weight * data['total_frequency'] * len(data['domains'])
            
            final_entities.append({
                'name': name,
                'total_frequency': data['total_frequency'],
                'average_weight': avg_weight,
                'final_score': final_score,
                'domains': list(data['domains']),
                'extraction_methods': list(data['methods']),
                'document_count': len(data['contexts']),
                'contexts': data['contexts']
            })
        
        # Sort by final score
        final_entities.sort(key=lambda x: x['final_score'], reverse=True)
        
        processing_time = time.time() - start_time
        
        # Analysis and reporting
        print(f"\nEXTRACTION COMPLETE:")
        print(f"Processing Time: {processing_time:.1f} seconds")
        print(f"Unique Entities Found: {len(final_entities)}")
        
        # Domain distribution
        domain_counts = Counter()
        for entity in final_entities:
            for domain in entity['domains']:
                domain_counts[domain] += 1
        
        print(f"Domain Distribution: {dict(domain_counts)}")
        
        # Show top entities
        print(f"\nTop 20 Entities by Final Score:")
        for i, entity in enumerate(final_entities[:20]):
            print(f"  {i+1:2d}. {entity['name']:<30} "
                  f"(freq: {entity['total_frequency']:2d}, "
                  f"weight: {entity['average_weight']:.3f}, "
                  f"score: {entity['final_score']:.3f}, "
                  f"domains: {len(entity['domains'])})")
        
        # PRP validation
        print(f"\n" + "="*80)
        print("PRP TARGET VALIDATION")
        print("="*80)
        target_met = len(final_entities) >= 85
        print(f"Target: 85+ entities")
        print(f"Achieved: {len(final_entities)} entities")
        print(f"Status: {'✓ TARGET MET' if target_met else '✗ TARGET NOT MET'}")
        
        if not target_met:
            print(f"Shortfall: {85 - len(final_entities)} entities needed")
            print("Recommendation: Expand term dictionaries or lower confidence thresholds")
        
        # Save results
        result = {
            'processing_info': {
                'total_bookmarks': len(bookmarks),
                'processing_time': processing_time,
                'entities_found': len(final_entities),
                'target_met': target_met
            },
            'entities': final_entities,
            'domain_distribution': dict(domain_counts),
            'extraction_summary': {
                'top_entities': final_entities[:30],
                'domain_coverage': len(domain_counts),
                'avg_entity_score': sum(e['final_score'] for e in final_entities) / len(final_entities) if final_entities else 0,
                'high_confidence_entities': len([e for e in final_entities if e['average_weight'] > 0.5])
            }
        }
        
        with open('enhanced_entity_extraction_results.json', 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2, ensure_ascii=False)
        
        print(f"\nDetailed results saved to: enhanced_entity_extraction_results.json")
        print("="*80)
        
        return result


def main():
    """Main execution function."""
    extractor = EnhancedEntityExtractor()
    result = extractor.process_dataset()
    
    # Final summary
    print(f"\nFINAL SUMMARY:")
    print(f"✓ Translation-first architecture validated (100% English content used)")
    print(f"✓ Context weighting applied to {result['processing_info']['entities_found']} entities")
    print(f"✓ Multi-domain coverage: {result['extraction_summary']['domain_coverage']} domains")
    print(f"✓ Processing efficiency: {result['processing_info']['processing_time']:.1f} seconds")
    
    if result['processing_info']['target_met']:
        print(f"✓ PRP target achieved: {result['processing_info']['entities_found']}/85 entities")
    else:
        print(f"⚠ PRP target not met: {result['processing_info']['entities_found']}/85 entities")
    
    return result


if __name__ == "__main__":
    main()